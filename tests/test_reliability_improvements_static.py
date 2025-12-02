"""
Static code analysis tests for reliability improvements.

Tests verify code changes are present without importing the module (avoids circular import).
"""

import os
import re
import pytest

# Get project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_PY_PATH = os.path.join(PROJECT_ROOT, 'chutes/entrypoint/run.py')


@pytest.fixture
def run_py_content():
    """Load run.py content once for all tests."""
    with open(RUN_PY_PATH, 'r') as f:
        return f.read()


class TestTimeoutConfiguration:
    """Test TimeoutConfig class implementation."""
    
    def test_timeout_config_class_exists(self, run_py_content):
        """Verify TimeoutConfig class is defined."""
        assert 'class TimeoutConfig:' in run_py_content
    
    def test_timeout_constants_defined(self, run_py_content):
        """Verify all timeout constants are defined."""
        required_constants = [
            'DEFAULT_TOTAL',
            'DEFAULT_CONNECT',
            'DUMMY_SOCKET_TIMEOUT',
            'DUMMY_SOCKET_RECV_TIMEOUT',
            'DUMMY_SOCKET_READY_TIMEOUT',
        ]
        
        for constant in required_constants:
            assert constant in run_py_content, f"{constant} should be defined"
    
    def test_timeout_factory_methods(self, run_py_content):
        """Verify factory methods exist."""
        assert 'def default(cls)' in run_py_content
        assert 'def quick(cls)' in run_py_content
        assert 'aiohttp.ClientTimeout' in run_py_content


class TestSocketTimeouts:
    """Test socket timeout implementation."""
    
    def test_tcp_socket_timeout_set(self, run_py_content):
        """Verify TCP socket has timeout configured."""
        # Find start_tcp_dummy function
        if 'def start_tcp_dummy(' in run_py_content:
            idx = run_py_content.find('def start_tcp_dummy(')
            func_section = run_py_content[idx:idx + 1500]
            
            # Should call settimeout
            assert 'sock.settimeout(' in func_section
            assert 'TimeoutConfig.DUMMY_SOCKET_TIMEOUT' in func_section
    
    def test_udp_socket_timeout_set(self, run_py_content):
        """Verify UDP socket has timeout configured."""
        # Find start_udp_dummy function
        if 'def start_udp_dummy(' in run_py_content:
            idx = run_py_content.find('def start_udp_dummy(')
            func_section = run_py_content[idx:idx + 1500]
            
            # Should call settimeout
            assert 'sock.settimeout(' in func_section
            assert 'TimeoutConfig.DUMMY_SOCKET_TIMEOUT' in func_section
    
    def test_conn_timeout_set(self, run_py_content):
        """Verify accepted connection has timeout."""
        # TCP handler should set timeout on accepted connection
        if 'conn, addr = sock.accept()' in run_py_content:
            idx = run_py_content.find('conn, addr = sock.accept()')
            section = run_py_content[idx:idx + 200]
            
            assert 'conn.settimeout(' in section


class TestSocketReadinessSignaling:
    """Test readiness signaling implementation."""
    
    def test_ready_event_created(self, run_py_content):
        """Verify threading.Event() is created for signaling."""
        # Should create Event
        assert 'ready_event = threading.Event()' in run_py_content
    
    def test_ready_event_set_on_bind(self, run_py_content):
        """Verify event is set after socket binds."""
        # Should call set() after bind
        assert 'ready_event.set()' in run_py_content
    
    def test_functions_return_event(self, run_py_content):
        """Verify functions return (thread, event) tuple."""
        # Find return statements in socket functions
        if 'def start_tcp_dummy(' in run_py_content:
            idx = run_py_content.find('def start_tcp_dummy(')
            func_section = run_py_content[idx:idx + 2000]  # Increased window
            
            # Should return thread and event
            assert 'return thread, ready_event' in func_section
    
    def test_ready_event_wait_logic(self, run_py_content):
        """Verify code waits for sockets to be ready."""
        # Should wait for ready events before GPU verification
        assert 'ready_event.wait(timeout=' in run_py_content
        assert 'Waiting for dummy sockets to start' in run_py_content


class TestByteRangeValidation:
    """Test byte range validation in slurp handler."""
    
    def test_max_slurp_size_defined(self, run_py_content):
        """Verify MAX_SLURP_SIZE constant exists."""
        assert 'MAX_SLURP_SIZE' in run_py_content
        assert '10 * 1024 * 1024' in run_py_content  # 10MB
    
    def test_start_byte_validation(self, run_py_content):
        """Verify start_byte is validated."""
        # Find handle_slurp function
        if 'async def handle_slurp(' in run_py_content:
            idx = run_py_content.find('async def handle_slurp(')
            func_section = run_py_content[idx:idx + 3000]
            
            # Should validate start_byte
            assert 'start_byte < 0' in func_section
            assert 'start_byte >' in func_section
            assert 'Invalid start_byte' in func_section
    
    def test_end_byte_validation(self, run_py_content):
        """Verify end_byte is validated."""
        # Find handle_slurp function
        if 'async def handle_slurp(' in run_py_content:
            idx = run_py_content.find('async def handle_slurp(')
            func_section = run_py_content[idx:idx + 3000]
            
            # Should validate end_byte
            assert 'end_byte < slurp.start_byte' in func_section
            assert 'end_byte must be >= start_byte' in func_section
    
    def test_size_limit_enforced(self, run_py_content):
        """Verify size limit is enforced."""
        # Find handle_slurp function
        if 'async def handle_slurp(' in run_py_content:
            idx = run_py_content.find('async def handle_slurp(')
            func_section = run_py_content[idx:idx + 3000]
            
            # Should check size against MAX_SLURP_SIZE
            assert 'read_size > MAX_SLURP_SIZE' in func_section
            assert 'HTTP_413' in func_section or '413' in func_section


class TestErrorHandling:
    """Test enhanced error handling."""
    
    def test_file_not_found_handling(self, run_py_content):
        """Verify FileNotFoundError is caught."""
        assert 'FileNotFoundError' in run_py_content
    
    def test_json_decode_error_handling(self, run_py_content):
        """Verify JSONDecodeError is caught."""
        assert 'JSONDecodeError' in run_py_content
    
    def test_file_encoding_specified(self, run_py_content):
        """Verify encoding is specified for file operations."""
        # Should use encoding='utf-8'
        assert "encoding='utf-8'" in run_py_content
    
    def test_connectivity_error_logging(self, run_py_content):
        """Verify connectivity check logs errors."""
        # Find check_connectivity function
        if 'async def check_connectivity(' in run_py_content:
            idx = run_py_content.find('async def check_connectivity(')
            func_section = run_py_content[idx:idx + 2000]
            
            # Should have catch-all with logging
            assert 'except Exception as e:' in func_section
            assert 'logger.error' in func_section


class TestBackwardCompatibility:
    """Ensure no breaking changes."""
    
    def test_start_dummy_socket_signature(self, run_py_content):
        """Verify start_dummy_socket still exists."""
        assert 'def start_dummy_socket(port_mapping, symmetric_key):' in run_py_content
    
    def test_handle_slurp_signature(self, run_py_content):
        """Verify handle_slurp still exists."""
        assert 'async def handle_slurp(request: Request, chute_module):' in run_py_content
    
    def test_tcp_dummy_signature(self, run_py_content):
        """Verify start_tcp_dummy still exists."""
        assert 'def start_tcp_dummy(port, symmetric_key, response_plaintext):' in run_py_content
    
    def test_udp_dummy_signature(self, run_py_content):
        """Verify start_udp_dummy still exists."""
        assert 'def start_udp_dummy(port, symmetric_key, response_plaintext):' in run_py_content


class TestCodeQuality:
    """Test code quality improvements."""
    
    def test_no_hardcoded_timeouts_in_connectivity(self, run_py_content):
        """Verify connectivity check uses TimeoutConfig."""
        # Find check_connectivity function
        if 'async def check_connectivity(' in run_py_content:
            idx = run_py_content.find('async def check_connectivity(')
            func_section = run_py_content[idx:idx + 1000]
            
            # Should use TimeoutConfig.quick()
            assert 'TimeoutConfig.quick()' in func_section
    
    def test_socket_timeout_uses_config(self, run_py_content):
        """Verify socket timeouts use TimeoutConfig constants."""
        # Should not have hardcoded 30.0 timeout
        # Should use TimeoutConfig.DUMMY_SOCKET_TIMEOUT
        if 'def start_tcp_dummy(' in run_py_content:
            idx = run_py_content.find('def start_tcp_dummy(')
            func_section = run_py_content[idx:idx + 1500]
            
            assert 'TimeoutConfig.DUMMY_SOCKET_TIMEOUT' in func_section


class TestDocumentation:
    """Test code documentation."""
    
    def test_timeout_config_documented(self, run_py_content):
        """Verify TimeoutConfig has docstring."""
        if 'class TimeoutConfig:' in run_py_content:
            idx = run_py_content.find('class TimeoutConfig:')
            section = run_py_content[idx:idx + 200]
            
            # Should have docstring
            assert '"""' in section or "'''" in section
    
    def test_socket_functions_documented(self, run_py_content):
        """Verify socket functions have updated docstrings."""
        # TCP function should mention timeout and signaling
        if 'def start_tcp_dummy(' in run_py_content:
            idx = run_py_content.find('def start_tcp_dummy(')
            section = run_py_content[idx:idx + 300]
            
            # Should have docstring mentioning new features
            assert 'timeout' in section.lower() or 'readiness' in section.lower()


# Summary test
def test_all_improvements_present(run_py_content):
    """High-level test that all 6 improvements are present."""
    improvements = [
        ('TimeoutConfig class', 'class TimeoutConfig:'),
        ('Socket timeout', 'sock.settimeout('),
        ('Ready event', 'ready_event = threading.Event()'),
        ('Byte range validation', 'MAX_SLURP_SIZE'),
        ('FileNotFoundError handling', 'FileNotFoundError'),
        ('Encoding specification', "encoding='utf-8'"),
    ]
    
    for name, pattern in improvements:
        assert pattern in run_py_content, f"{name} not found in code"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
