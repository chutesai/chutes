"""
Custom exceptions for the Chutes platform.

This module defines a hierarchy of exceptions used throughout the codebase
to provide clear error handling and better debugging information.
"""


class ChutesError(Exception):
    """
    Base exception for all Chutes-related errors.
    
    All custom exceptions in the Chutes SDK should inherit from this class
    to allow for catch-all error handling when needed.
    """
    pass


class ConfigurationError(ChutesError):
    """
    Raised when there's an issue with configuration.
    
    Examples:
        - Missing configuration file
        - Invalid configuration values
        - Environment variable issues
    """
    pass


class AuthenticationError(ChutesError):
    """
    Raised when authentication fails or is missing.
    
    Examples:
        - Missing credentials
        - Invalid signature
        - Expired tokens
    """
    pass


class DeploymentError(ChutesError):
    """
    Raised when deployment operations fail.
    
    Examples:
        - Image build failures
        - Chute deployment failures
        - Upload errors
    """
    pass


class ValidationError(ChutesError):
    """
    Raised when input validation fails.
    
    Examples:
        - Invalid path format
        - Invalid port number
        - Schema validation errors
    """
    pass


class UserAbortedError(ChutesError):
    """
    Raised when user cancels an operation.
    
    This is used to distinguish between errors and intentional cancellations,
    allowing for graceful exit without error messages.
    """
    pass


class NetworkError(ChutesError):
    """
    Raised when network operations fail.
    
    Examples:
        - Connection timeouts
        - API unavailable
        - DNS resolution failures
    """
    pass


# Legacy exceptions - kept for backward compatibility
class InvalidPath(ValidationError):
    """Invalid path specification."""
    pass


class DuplicatePath(ValidationError):
    """Duplicate path found."""
    pass


class AuthenticationRequired(AuthenticationError):
    """Authentication credentials are required but missing."""
    pass


class NotConfigured(ConfigurationError):
    """Configuration is incomplete or missing."""
    pass


class StillProvisioning(ChutesError):
    """Resource is still being provisioned and not ready yet."""
    pass
