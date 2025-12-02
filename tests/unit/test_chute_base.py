"""
Unit tests for the core Chute class functionality.
"""

import os
import uuid
import asyncio
import pytest
from unittest.mock import Mock, patch, AsyncMock
from chutes.chute import Chute, Cord, Job
from chutes.image import Image
from chutes.chute.node_selector import NodeSelector


@pytest.fixture
def basic_chute():
    """Create a basic Chute instance for testing."""
    return Chute(
        username="testuser",
        name="testchute",
        image="python:3.12",
        tagline="Test chute",
        readme="Test readme"
    )


@pytest.fixture
def chute_with_image():
    """Create a Chute instance with an Image object."""
    image = Image(username="testuser", name="testimage", tag="latest")
    return Chute(
        username="testuser",
        name="testchute",
        image=image,
        tagline="Test chute with image",
        readme="Test readme"
    )


@pytest.fixture
def chute_with_node_selector():
    """Create a Chute instance with a NodeSelector."""
    node_selector = NodeSelector(
        gpu_count=2,
        min_vram_gb_per_gpu=24,
        include=["node1", "node2"],
        exclude=["node3"]
    )
    return Chute(
        username="testuser",
        name="testchute",
        image="python:3.12",
        node_selector=node_selector
    )


class TestChuteInitialization:
    """Test Chute object creation with various parameters."""

    def test_basic_initialization(self, basic_chute):
        """Test basic Chute initialization with minimal parameters."""
        assert basic_chute.name == "testchute"
        assert basic_chute.tagline == "Test chute"
        assert basic_chute.readme == "Test readme"
        assert basic_chute._username == "testuser"
        assert basic_chute._image == "python:3.12"

    def test_initialization_with_image_object(self, chute_with_image):
        """Test Chute initialization with an Image object."""
        assert chute_with_image.name == "testchute"
        assert isinstance(chute_with_image.image, Image)
        assert chute_with_image.image.name == "testimage"

    def test_initialization_with_node_selector(self, chute_with_node_selector):
        """Test Chute initialization with NodeSelector."""
        assert chute_with_node_selector.node_selector is not None
        assert chute_with_node_selector.node_selector.gpu_count == 2
        assert chute_with_node_selector.node_selector.min_vram_gb_per_gpu == 24

    def test_initialization_with_all_parameters(self):
        """Test Chute initialization with all available parameters."""
        node_selector = NodeSelector(gpu_count=1, min_vram_gb_per_gpu=16)
        chute = Chute(
            username="testuser",
            name="fullchute",
            image="python:3.12",
            tagline="Full test",
            readme="Full readme",
            standard_template="template1",
            revision="v1.0",
            node_selector=node_selector,
            concurrency=5,
            max_instances=10,
            shutdown_after_seconds=600,
            scaling_threshold=0.8,
            allow_external_egress=True,
            encrypted_fs=True,
            passthrough_headers={"X-Custom": "value"}
        )
        
        assert chute.name == "fullchute"
        assert chute.concurrency == 5
        assert chute.max_instances == 10
        assert chute.shutdown_after_seconds == 600
        assert chute.scaling_threshold == 0.8
        assert chute.allow_external_egress is True
        assert chute.encrypted_fs is True
        assert chute.passthrough_headers == {"X-Custom": "value"}
        assert chute.revision == "v1.0"
        assert chute.standard_template == "template1"

    def test_initialization_defaults(self):
        """Test that default values are properly set."""
        chute = Chute(username="user", name="chute", image="python:3.12")
        
        assert chute.tagline == ""
        assert chute.readme == ""
        assert chute.concurrency == 1
        assert chute.max_instances == 1
        assert chute.shutdown_after_seconds == 300
        assert chute.scaling_threshold == 0.75
        assert chute.allow_external_egress is False
        assert chute.encrypted_fs is False
        assert chute.passthrough_headers == {}
        assert chute.docs_url is None
        assert chute.redoc_url is None

    def test_initialization_creates_empty_lists(self, basic_chute):
        """Test that initialization creates empty lists for hooks, cords, and jobs."""
        assert basic_chute._startup_hooks == []
        assert basic_chute._shutdown_hooks == []
        assert basic_chute._cords == []
        assert basic_chute._jobs == []


class TestChuteUidGeneration:
    """Test UUID generation for Chute instances."""

    def test_uid_generation_consistency(self):
        """Test that UID is consistent for same username/name combination."""
        chute1 = Chute(username="user1", name="chute1", image="python:3.12")
        chute2 = Chute(username="user1", name="chute1", image="python:3.12")
        
        assert chute1.uid == chute2.uid

    def test_uid_generation_different_usernames(self):
        """Test that different usernames produce different UIDs."""
        chute1 = Chute(username="user1", name="chute1", image="python:3.12")
        chute2 = Chute(username="user2", name="chute1", image="python:3.12")
        
        assert chute1.uid != chute2.uid

    def test_uid_generation_different_names(self):
        """Test that different names produce different UIDs."""
        chute1 = Chute(username="user1", name="chute1", image="python:3.12")
        chute2 = Chute(username="user1", name="chute2", image="python:3.12")
        
        assert chute1.uid != chute2.uid

    def test_uid_is_valid_uuid(self, basic_chute):
        """Test that generated UID is a valid UUID string."""
        try:
            uuid.UUID(basic_chute.uid)
        except ValueError:
            pytest.fail("UID is not a valid UUID")

    def test_uid_uses_namespace_oid(self):
        """Test that UID generation uses the correct namespace."""
        chute = Chute(username="testuser", name="testchute", image="python:3.12")
        expected_uid = str(uuid.uuid5(uuid.NAMESPACE_OID, "testuser::chute::testchute"))
        
        assert chute.uid == expected_uid


class TestChuteProperties:
    """Test all property getters."""

    def test_name_property(self, basic_chute):
        """Test name property getter."""
        assert basic_chute.name == "testchute"

    def test_readme_property(self, basic_chute):
        """Test readme property getter."""
        assert basic_chute.readme == "Test readme"

    def test_tagline_property(self, basic_chute):
        """Test tagline property getter."""
        assert basic_chute.tagline == "Test chute"

    def test_uid_property(self, basic_chute):
        """Test uid property getter."""
        assert isinstance(basic_chute.uid, str)
        assert len(basic_chute.uid) == 36  # UUID string length

    def test_image_property_string(self, basic_chute):
        """Test image property getter with string image."""
        assert basic_chute.image == "python:3.12"

    def test_image_property_object(self, chute_with_image):
        """Test image property getter with Image object."""
        assert isinstance(chute_with_image.image, Image)

    def test_cords_property(self, basic_chute):
        """Test cords property getter."""
        assert basic_chute.cords == []
        assert isinstance(basic_chute.cords, list)

    def test_jobs_property(self, basic_chute):
        """Test jobs property getter."""
        assert basic_chute.jobs == []
        assert isinstance(basic_chute.jobs, list)

    def test_node_selector_property(self, chute_with_node_selector):
        """Test node_selector property getter."""
        assert chute_with_node_selector.node_selector is not None
        assert isinstance(chute_with_node_selector.node_selector, NodeSelector)

    def test_node_selector_property_none(self, basic_chute):
        """Test node_selector property when not set."""
        assert basic_chute.node_selector is None

    def test_standard_template_property(self):
        """Test standard_template property getter."""
        chute = Chute(
            username="user",
            name="chute",
            image="python:3.12",
            standard_template="template1"
        )
        assert chute.standard_template == "template1"


class TestStartupHookRegistration:
    """Test on_startup() decorator registration."""

    def test_startup_hook_registration(self, basic_chute):
        """Test that startup hooks are properly registered."""
        @basic_chute.on_startup()
        def startup_func(app):
            pass
        
        assert len(basic_chute._startup_hooks) == 1

    def test_startup_hook_with_priority(self, basic_chute):
        """Test startup hook registration with custom priority."""
        @basic_chute.on_startup(priority=10)
        def startup_func(app):
            pass
        
        assert len(basic_chute._startup_hooks) == 1
        priority, func = basic_chute._startup_hooks[0]
        assert priority == 10

    def test_startup_hook_default_priority(self, basic_chute):
        """Test that default priority is 50."""
        @basic_chute.on_startup()
        def startup_func(app):
            pass
        
        priority, func = basic_chute._startup_hooks[0]
        assert priority == 50

    def test_multiple_startup_hooks(self, basic_chute):
        """Test registering multiple startup hooks."""
        @basic_chute.on_startup(priority=10)
        def startup_func1(app):
            pass
        
        @basic_chute.on_startup(priority=20)
        def startup_func2(app):
            pass
        
        @basic_chute.on_startup(priority=30)
        def startup_func3(app):
            pass
        
        assert len(basic_chute._startup_hooks) == 3

    def test_async_startup_hook_registration(self, basic_chute):
        """Test that async startup hooks are properly registered."""
        @basic_chute.on_startup()
        async def async_startup_func(app):
            pass
        
        assert len(basic_chute._startup_hooks) == 1


class TestShutdownHookRegistration:
    """Test on_shutdown() decorator registration."""

    def test_shutdown_hook_registration(self, basic_chute):
        """Test that shutdown hooks are properly registered."""
        @basic_chute.on_shutdown()
        def shutdown_func(app):
            pass
        
        assert len(basic_chute._shutdown_hooks) == 1

    def test_shutdown_hook_with_priority(self, basic_chute):
        """Test shutdown hook registration with custom priority."""
        @basic_chute.on_shutdown(priority=20)
        def shutdown_func(app):
            pass
        
        assert len(basic_chute._shutdown_hooks) == 1
        priority, func = basic_chute._shutdown_hooks[0]
        assert priority == 20

    def test_shutdown_hook_default_priority(self, basic_chute):
        """Test that default priority is 50."""
        @basic_chute.on_shutdown()
        def shutdown_func(app):
            pass
        
        priority, func = basic_chute._shutdown_hooks[0]
        assert priority == 50

    def test_multiple_shutdown_hooks(self, basic_chute):
        """Test registering multiple shutdown hooks."""
        @basic_chute.on_shutdown(priority=10)
        def shutdown_func1(app):
            pass
        
        @basic_chute.on_shutdown(priority=20)
        def shutdown_func2(app):
            pass
        
        assert len(basic_chute._shutdown_hooks) == 2

    def test_async_shutdown_hook_registration(self, basic_chute):
        """Test that async shutdown hooks are properly registered."""
        @basic_chute.on_shutdown()
        async def async_shutdown_func(app):
            pass
        
        assert len(basic_chute._shutdown_hooks) == 1


class TestStartupHookPriorityOrdering:
    """Test that startup hooks execute in correct priority order."""

    @pytest.mark.asyncio
    async def test_startup_hooks_execute_in_priority_order(self, basic_chute):
        """Test that startup hooks execute in ascending priority order."""
        execution_order = []
        
        @basic_chute.on_startup(priority=30)
        def startup_func3(app):
            execution_order.append(3)
        
        @basic_chute.on_startup(priority=10)
        def startup_func1(app):
            execution_order.append(1)
        
        @basic_chute.on_startup(priority=20)
        def startup_func2(app):
            execution_order.append(2)
        
        with patch('chutes.chute.base.is_remote', return_value=True):
            await basic_chute.initialize()
        
        assert execution_order == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_startup_hooks_with_same_priority(self, basic_chute):
        """Test that hooks with same priority maintain registration order."""
        execution_order = []
        
        @basic_chute.on_startup(priority=50)
        def startup_func1(app):
            execution_order.append(1)
        
        @basic_chute.on_startup(priority=50)
        def startup_func2(app):
            execution_order.append(2)
        
        with patch('chutes.chute.base.is_remote', return_value=True):
            await basic_chute.initialize()
        
        assert execution_order == [1, 2]

    @pytest.mark.asyncio
    async def test_startup_hooks_priority_range(self, basic_chute):
        """Test hooks with wide priority range."""
        execution_order = []
        
        @basic_chute.on_startup(priority=100)
        def startup_late(app):
            execution_order.append('late')
        
        @basic_chute.on_startup(priority=0)
        def startup_early(app):
            execution_order.append('early')
        
        @basic_chute.on_startup(priority=50)
        def startup_middle(app):
            execution_order.append('middle')
        
        with patch('chutes.chute.base.is_remote', return_value=True):
            await basic_chute.initialize()
        
        assert execution_order == ['early', 'middle', 'late']


class TestShutdownHookPriorityOrdering:
    """Test that shutdown hooks respect priority order."""

    def test_shutdown_hooks_stored_with_priority(self, basic_chute):
        """Test that shutdown hooks are stored with their priority values."""
        @basic_chute.on_shutdown(priority=10)
        def shutdown_func1(app):
            pass
        
        @basic_chute.on_shutdown(priority=90)
        def shutdown_func2(app):
            pass
        
        priorities = [priority for priority, func in basic_chute._shutdown_hooks]
        assert priorities == [10, 90]

    def test_shutdown_hooks_can_be_sorted(self, basic_chute):
        """Test that shutdown hooks can be sorted by priority."""
        @basic_chute.on_shutdown(priority=50)
        def shutdown_func1(app):
            pass
        
        @basic_chute.on_shutdown(priority=10)
        def shutdown_func2(app):
            pass
        
        @basic_chute.on_shutdown(priority=30)
        def shutdown_func3(app):
            pass
        
        sorted_hooks = sorted(basic_chute._shutdown_hooks, key=lambda x: x[0])
        priorities = [priority for priority, func in sorted_hooks]
        assert priorities == [10, 30, 50]


class TestAsyncStartupHooks:
    """Test async startup hook execution."""

    @pytest.mark.asyncio
    async def test_async_startup_hook_execution(self, basic_chute):
        """Test that async startup hooks execute properly."""
        executed = []
        
        @basic_chute.on_startup()
        async def async_startup(app):
            await asyncio.sleep(0.01)
            executed.append('async')
        
        with patch('chutes.chute.base.is_remote', return_value=True):
            await basic_chute.initialize()
        
        assert 'async' in executed

    @pytest.mark.asyncio
    async def test_multiple_async_startup_hooks(self, basic_chute):
        """Test multiple async startup hooks execute in order."""
        execution_order = []
        
        @basic_chute.on_startup(priority=10)
        async def async_startup1(app):
            await asyncio.sleep(0.01)
            execution_order.append(1)
        
        @basic_chute.on_startup(priority=20)
        async def async_startup2(app):
            await asyncio.sleep(0.01)
            execution_order.append(2)
        
        with patch('chutes.chute.base.is_remote', return_value=True):
            await basic_chute.initialize()
        
        assert execution_order == [1, 2]

    @pytest.mark.asyncio
    async def test_async_startup_hook_receives_app(self, basic_chute):
        """Test that async startup hooks receive the app instance."""
        received_app = None
        
        @basic_chute.on_startup()
        async def async_startup(app):
            nonlocal received_app
            received_app = app
        
        with patch('chutes.chute.base.is_remote', return_value=True):
            await basic_chute.initialize()
        
        assert received_app is basic_chute


class TestSyncStartupHooks:
    """Test synchronous startup hook execution."""

    @pytest.mark.asyncio
    async def test_sync_startup_hook_execution(self, basic_chute):
        """Test that sync startup hooks execute properly."""
        executed = []
        
        @basic_chute.on_startup()
        def sync_startup(app):
            executed.append('sync')
        
        with patch('chutes.chute.base.is_remote', return_value=True):
            await basic_chute.initialize()
        
        assert 'sync' in executed

    @pytest.mark.asyncio
    async def test_multiple_sync_startup_hooks(self, basic_chute):
        """Test multiple sync startup hooks execute in order."""
        execution_order = []
        
        @basic_chute.on_startup(priority=10)
        def sync_startup1(app):
            execution_order.append(1)
        
        @basic_chute.on_startup(priority=20)
        def sync_startup2(app):
            execution_order.append(2)
        
        with patch('chutes.chute.base.is_remote', return_value=True):
            await basic_chute.initialize()
        
        assert execution_order == [1, 2]

    @pytest.mark.asyncio
    async def test_sync_startup_hook_receives_app(self, basic_chute):
        """Test that sync startup hooks receive the app instance."""
        received_app = None
        
        @basic_chute.on_startup()
        def sync_startup(app):
            nonlocal received_app
            received_app = app
        
        with patch('chutes.chute.base.is_remote', return_value=True):
            await basic_chute.initialize()
        
        assert received_app is basic_chute

    @pytest.mark.asyncio
    async def test_mixed_sync_async_startup_hooks(self, basic_chute):
        """Test that sync and async hooks can be mixed."""
        execution_order = []
        
        @basic_chute.on_startup(priority=10)
        def sync_startup(app):
            execution_order.append('sync')
        
        @basic_chute.on_startup(priority=20)
        async def async_startup(app):
            await asyncio.sleep(0.01)
            execution_order.append('async')
        
        with patch('chutes.chute.base.is_remote', return_value=True):
            await basic_chute.initialize()
        
        assert execution_order == ['sync', 'async']


class TestCordRegistration:
    """Test that cords are properly added to the chute."""

    def test_cord_decorator_creates_cord(self, basic_chute):
        """Test that cord decorator creates a Cord instance."""
        @basic_chute.cord()
        def test_cord(input_data):
            return {"result": "success"}
        
        assert len(basic_chute._cords) == 1
        assert isinstance(basic_chute._cords[0], Cord)

    def test_multiple_cord_registration(self, basic_chute):
        """Test registering multiple cords."""
        @basic_chute.cord()
        def cord1(input_data):
            return {"result": "1"}
        
        @basic_chute.cord()
        def cord2(input_data):
            return {"result": "2"}
        
        @basic_chute.cord()
        def cord3(input_data):
            return {"result": "3"}
        
        assert len(basic_chute._cords) == 3

    def test_cord_with_parameters(self, basic_chute):
        """Test cord registration with parameters."""
        @basic_chute.cord(path="/custom", stream=True)
        def test_cord(input_data):
            return {"result": "success"}
        
        assert len(basic_chute._cords) == 1
        cord = basic_chute._cords[0]
        assert cord._stream is True

    def test_cords_property_returns_list(self, basic_chute):
        """Test that cords property returns the correct list."""
        @basic_chute.cord()
        def test_cord(input_data):
            return {"result": "success"}
        
        cords = basic_chute.cords
        assert isinstance(cords, list)
        assert len(cords) == 1
        assert isinstance(cords[0], Cord)


class TestJobRegistration:
    """Test that jobs are properly added to the chute."""

    def test_job_decorator_creates_job(self, basic_chute):
        """Test that job decorator creates a Job instance."""
        @basic_chute.job()
        def test_job():
            pass
        
        assert len(basic_chute._jobs) == 1
        assert isinstance(basic_chute._jobs[0], Job)

    def test_multiple_job_registration(self, basic_chute):
        """Test registering multiple jobs."""
        @basic_chute.job()
        def job1():
            pass
        
        @basic_chute.job()
        def job2():
            pass
        
        assert len(basic_chute._jobs) == 2

    def test_job_with_parameters(self, basic_chute):
        """Test job registration with parameters."""
        @basic_chute.job(timeout=300, ssh=True)
        def test_job():
            pass
        
        assert len(basic_chute._jobs) == 1
        job = basic_chute._jobs[0]
        assert job._timeout == 300
        assert job._ssh is True

    def test_jobs_property_returns_list(self, basic_chute):
        """Test that jobs property returns the correct list."""
        @basic_chute.job()
        def test_job():
            pass
        
        jobs = basic_chute.jobs
        assert isinstance(jobs, list)
        assert len(jobs) == 1
        assert isinstance(jobs[0], Job)


class TestNoProxyEnvironmentSetup:
    """Test NO_PROXY environment variable configuration."""

    def test_no_proxy_set_in_remote_context(self):
        """Test that NO_PROXY is set when CHUTES_EXECUTION_CONTEXT is REMOTE."""
        # Save original environment
        original_context = os.environ.get("CHUTES_EXECUTION_CONTEXT")
        original_no_proxy = os.environ.get("NO_PROXY")
        
        try:
            # Set remote context before importing
            os.environ["CHUTES_EXECUTION_CONTEXT"] = "REMOTE"
            if "NO_PROXY" in os.environ:
                del os.environ["NO_PROXY"]
            
            # Re-import the module to trigger the environment setup
            import importlib
            import chutes.chute.base
            importlib.reload(chutes.chute.base)
            
            # Check NO_PROXY is set
            no_proxy = os.environ.get("NO_PROXY")
            assert no_proxy is not None
            assert "localhost" in no_proxy
            assert "127.0.0.1" in no_proxy
            assert "api" in no_proxy
            assert "api.chutes.svc" in no_proxy
            assert "api.chutes.svc.cluster.local" in no_proxy
        
        finally:
            # Restore original environment
            if original_context:
                os.environ["CHUTES_EXECUTION_CONTEXT"] = original_context
            elif "CHUTES_EXECUTION_CONTEXT" in os.environ:
                del os.environ["CHUTES_EXECUTION_CONTEXT"]
            
            if original_no_proxy:
                os.environ["NO_PROXY"] = original_no_proxy
            elif "NO_PROXY" in os.environ:
                del os.environ["NO_PROXY"]
            
            # Reload module to restore original state
            import chutes.chute.base
            importlib.reload(chutes.chute.base)

    def test_no_proxy_preserves_existing_values(self):
        """Test that existing NO_PROXY values are preserved."""
        # Save original environment
        original_context = os.environ.get("CHUTES_EXECUTION_CONTEXT")
        original_no_proxy = os.environ.get("NO_PROXY")
        
        try:
            # Set remote context and existing NO_PROXY
            os.environ["CHUTES_EXECUTION_CONTEXT"] = "REMOTE"
            os.environ["NO_PROXY"] = "existing.domain.com"
            
            # Re-import the module
            import importlib
            import chutes.chute.base
            importlib.reload(chutes.chute.base)
            
            # Check that existing value is preserved
            no_proxy = os.environ.get("NO_PROXY")
            assert "existing.domain.com" in no_proxy
            assert "localhost" in no_proxy
        
        finally:
            # Restore original environment
            if original_context:
                os.environ["CHUTES_EXECUTION_CONTEXT"] = original_context
            elif "CHUTES_EXECUTION_CONTEXT" in os.environ:
                del os.environ["CHUTES_EXECUTION_CONTEXT"]
            
            if original_no_proxy:
                os.environ["NO_PROXY"] = original_no_proxy
            elif "NO_PROXY" in os.environ:
                del os.environ["NO_PROXY"]
            
            # Reload module
            import chutes.chute.base
            importlib.reload(chutes.chute.base)

    def test_no_proxy_not_set_in_non_remote_context(self):
        """Test that NO_PROXY is not modified when not in REMOTE context."""
        # Save original environment
        original_context = os.environ.get("CHUTES_EXECUTION_CONTEXT")
        original_no_proxy = os.environ.get("NO_PROXY")
        
        try:
            # Set non-remote context
            os.environ["CHUTES_EXECUTION_CONTEXT"] = "LOCAL"
            if "NO_PROXY" in os.environ:
                del os.environ["NO_PROXY"]
            
            # Re-import the module
            import importlib
            import chutes.chute.base
            importlib.reload(chutes.chute.base)
            
            # Check NO_PROXY is not set
            no_proxy = os.environ.get("NO_PROXY")
            assert no_proxy is None
        
        finally:
            # Restore original environment
            if original_context:
                os.environ["CHUTES_EXECUTION_CONTEXT"] = original_context
            elif "CHUTES_EXECUTION_CONTEXT" in os.environ:
                del os.environ["CHUTES_EXECUTION_CONTEXT"]
            
            if original_no_proxy:
                os.environ["NO_PROXY"] = original_no_proxy
            elif "NO_PROXY" in os.environ:
                del os.environ["NO_PROXY"]
            
            # Reload module
            import chutes.chute.base
            importlib.reload(chutes.chute.base)


class TestChuteInitializeMethod:
    """Test the initialize() method behavior."""

    @pytest.mark.asyncio
    async def test_initialize_skips_when_not_remote(self, basic_chute):
        """Test that initialize() returns early when not in remote context."""
        executed = []
        
        @basic_chute.on_startup()
        def startup_func(app):
            executed.append('startup')
        
        with patch('chutes.util.context.is_remote', return_value=False):
            await basic_chute.initialize()
        
        # Hook should not execute when not remote
        assert len(executed) == 0

    @pytest.mark.asyncio
    async def test_initialize_executes_hooks_when_remote(self, basic_chute):
        """Test that initialize() executes hooks when in remote context."""
        executed = []
        
        @basic_chute.on_startup()
        def startup_func(app):
            executed.append('startup')
        
        with patch('chutes.chute.base.is_remote', return_value=True):
            await basic_chute.initialize()
        
        assert 'startup' in executed

    @pytest.mark.asyncio
    async def test_initialize_adds_cord_routes(self, basic_chute):
        """Test that initialize() adds API routes for cords."""
        @basic_chute.cord()
        def test_cord(input_data):
            return {"result": "success"}
        
        with patch('chutes.chute.base.is_remote', return_value=True):
            with patch.object(basic_chute, 'add_api_route') as mock_add_route:
                await basic_chute.initialize()
                
                # Verify add_api_route was called
                assert mock_add_route.called

    @pytest.mark.asyncio
    async def test_initialize_logs_job_definitions(self, basic_chute):
        """Test that initialize() logs job definitions."""
        @basic_chute.job()
        def test_job():
            pass
        
        with patch('chutes.chute.base.is_remote', return_value=True):
            # Just verify it doesn't raise an error
            await basic_chute.initialize()
