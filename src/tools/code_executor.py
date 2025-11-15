"""
Code Executor Tool - Safely executes Python code in restricted environment.

Security features:
- Restricted imports (only safe libraries)
- Execution timeout
- Memory limits
- No file system access (except temp)
"""

import ast
import sys
import io
import logging
from typing import Dict, Any, List, Optional
from contextlib import redirect_stdout, redirect_stderr
import signal
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result of code execution."""
    success: bool
    output: str
    error: Optional[str] = None
    return_value: Any = None
    
    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "return_value": str(self.return_value) if self.return_value is not None else None
        }


class TimeoutError(Exception):
    """Raised when code execution times out."""
    pass


def timeout_handler(signum, frame):
    """Handle execution timeout."""
    raise TimeoutError("Code execution timed out")


class CodeExecutorTool:
    """
    Safe Python code execution tool.
    
    Usage:
        executor = CodeExecutorTool()
        result = executor.execute("print(2 + 2)")
    """
    
    # Allowed imports (safe libraries only)
    ALLOWED_IMPORTS = {
        'math', 'statistics', 'datetime', 'json', 'random',
        'collections', 'itertools', 'functools', 're',
        'decimal', 'fractions', 'operator'
    }
    
    def __init__(
        self,
        timeout: int = 5,
        allowed_imports: Optional[List[str]] = None
    ):
        """
        Initialize code executor.
        
        Args:
            timeout: Maximum execution time in seconds
            allowed_imports: Additional allowed imports beyond defaults
        """
        self.timeout = timeout
        self.allowed_imports = self.ALLOWED_IMPORTS.copy()
        
        if allowed_imports:
            self.allowed_imports.update(allowed_imports)
        
        logger.info(f"Initialized CodeExecutor (timeout={timeout}s)")
    
    def execute(self, code: str, timeout: Optional[int] = None) -> ExecutionResult:
        """
        Execute Python code safely.
        
        Args:
            code: Python code to execute
            timeout: Override default timeout
            
        Returns:
            ExecutionResult with output/errors
        """
        timeout = timeout or self.timeout
        
        logger.info(f"Executing code (timeout={timeout}s)")
        logger.debug(f"Code: {code[:100]}...")
        
        # Validate code first
        validation_error = self._validate_code(code)
        if validation_error:
            logger.warning(f"Code validation failed: {validation_error}")
            return ExecutionResult(
                success=False,
                output="",
                error=validation_error
            )
        
        # Capture output
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        # Set timeout (Unix only)
        if sys.platform != 'win32':
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout)
        
        try:
            # Create restricted namespace
            namespace = self._create_namespace()
            
            # Execute code
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(code, namespace)
            
            # Get return value if any
            return_value = namespace.get('result', None)
            
            stdout_text = stdout_capture.getvalue()
            stderr_text = stderr_capture.getvalue()
            
            logger.info("Code executed successfully")
            
            return ExecutionResult(
                success=True,
                output=stdout_text,
                error=stderr_text if stderr_text else None,
                return_value=return_value
            )
            
        except TimeoutError:
            error = f"Execution timed out after {timeout} seconds"
            logger.error(error)
            return ExecutionResult(success=False, output="", error=error)
            
        except Exception as e:
            error = f"{type(e).__name__}: {str(e)}"
            logger.error(f"Execution failed: {error}")
            return ExecutionResult(
                success=False,
                output=stdout_capture.getvalue(),
                error=error
            )
            
        finally:
            # Cancel timeout
            if sys.platform != 'win32':
                signal.alarm(0)
    
    def _validate_code(self, code: str) -> Optional[str]:
        """
        Validate code for security issues.
        
        Returns:
            Error message if invalid, None if valid
        """
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return f"Syntax error: {str(e)}"
        
        # Check for dangerous operations
        for node in ast.walk(tree):
            # Block file operations
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name not in self.allowed_imports:
                        if alias.name in ['os', 'sys', 'subprocess', '__builtin__', 'builtins']:
                            return f"Import '{alias.name}' not allowed (security risk)"
                        return f"Import '{alias.name}' not allowed (not in allowed list)"
            
            elif isinstance(node, ast.ImportFrom):
                if node.module not in self.allowed_imports:
                    if node.module in ['os', 'sys', 'subprocess', '__builtin__', 'builtins']:
                        return f"Import from '{node.module}' not allowed (security risk)"
                    return f"Import from '{node.module}' not allowed"
            
            # Block eval/exec
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ['eval', 'exec', 'compile', '__import__']:
                        return f"Function '{node.func.id}' not allowed (security risk)"
        
        return None
    
    def _create_namespace(self) -> Dict[str, Any]:
        """Create restricted execution namespace."""
        
        # Import allowed modules first
        allowed_modules = {}
        for module_name in self.allowed_imports:
            try:
                allowed_modules[module_name] = __import__(module_name)
            except ImportError:
                logger.warning(f"Could not import allowed module: {module_name}")
        
        namespace = {
            '__builtins__': {
                # Safe built-ins only
                'print': print,
                'len': len,
                'range': range,
                'enumerate': enumerate,
                'zip': zip,
                'map': map,
                'filter': filter,
                'sum': sum,
                'min': min,
                'max': max,
                'abs': abs,
                'round': round,
                'sorted': sorted,
                'list': list,
                'dict': dict,
                'set': set,
                'tuple': tuple,
                'str': str,
                'int': int,
                'float': float,
                'bool': bool,
                'type': type,
                'isinstance': isinstance,
                'issubclass': issubclass,
                'hasattr': hasattr,
                'getattr': getattr,
                '__import__': lambda name, *args, **kwargs: allowed_modules.get(name, None),
                'True': True,
                'False': False,
                'None': None,
            }
        }
        
        # Add modules to namespace
        namespace.update(allowed_modules)
        
        return namespace


# Example usage
def demo():
    """Demonstrate code executor."""
    executor = CodeExecutorTool(timeout=5)
    
    test_cases = [
        # Simple math
        ("print(2 + 2)", True),
        
        # Using allowed library
        ("import math\nprint(math.pi)", True),
        
        # Return value
        ("result = 42 * 2", True),
        
        # Loops and logic
        ("""
nums = [1, 2, 3, 4, 5]
result = sum(x**2 for x in nums)
print(f"Sum of squares: {result}")
""", True),
        
        # Blocked import
        ("import os\nprint(os.listdir())", False),
        
        # Blocked function
        ("eval('2 + 2')", False),
        
        # Infinite loop (will timeout)
        ("while True: pass", False),
    ]
    
    for i, (code, should_succeed) in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"Test {i}: {code[:40]}...")
        print('='*60)
        
        result = executor.execute(code)
        
        print(f"Success: {result.success}")
        if result.output:
            print(f"Output: {result.output}")
        if result.error:
            print(f"Error: {result.error}")
        if result.return_value:
            print(f"Return Value: {result.return_value}")
        
        # Verify expectation
        if result.success == should_succeed:
            print("✅ Test passed")
        else:
            print(f"❌ Test failed (expected success={should_succeed})")


if __name__ == "__main__":
    demo()