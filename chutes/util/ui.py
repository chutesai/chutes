"""
User interface utilities for CLI interactions.

This module provides helpers for user prompts, confirmations, and displaying
information in a consistent and user-friendly way.
"""

from typing import Optional, List
from rich.console import Console
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich import box
from loguru import logger


class UIHelper:
    """
    Helper class for CLI user interactions.
    
    Provides consistent UX for confirmations, prompts, and information display.
    
    Example:
        >>> ui = UIHelper()
        >>> if ui.confirm("Deploy this chute?"):
        ...     perform_deployment()
    """
    
    def __init__(self, console: Optional[Console] = None):
        """
        Initialize the UI helper.
        
        Args:
            console: Optional Rich console instance. If None, creates a new one.
        """
        self.console = console or Console()
    
    def confirm(
        self,
        message: str,
        default: bool = False,
        style: str = "bold underline"
    ) -> bool:
        """
        Prompt user for yes/no confirmation.
        
        Args:
            message: The confirmation message to display.
            default: Default value if user just presses Enter.
            style: Rich style string for the message.
            
        Returns:
            True if user confirmed, False otherwise.
            
        Example:
            >>> if ui.confirm("Continue with deployment?"):
            ...     deploy()
        """
        styled_message = f"[{style}]{message}[/{style}]"
        return Confirm.ask(styled_message, default=default, console=self.console)
    
    def prompt(
        self,
        message: str,
        default: Optional[str] = None,
        password: bool = False
    ) -> str:
        """
        Prompt user for text input.
        
        Args:
            message: The prompt message.
            default: Default value if user just presses Enter.
            password: If True, hide input (for passwords).
            
        Returns:
            User's input string.
        """
        return Prompt.ask(
            message,
            default=default,
            password=password,
            console=self.console
        )
    
    def show_file_list(
        self,
        files: List[str],
        title: str = "Files",
        max_display: int = 10,
        show_all_prompt: bool = True
    ) -> bool:
        """
        Display a list of files with optional truncation.
        
        Args:
            files: List of file paths to display.
            title: Title for the file list.
            max_display: Maximum number of files to show initially.
            show_all_prompt: If True, prompt to show remaining files.
            
        Returns:
            True if user confirmed (or no prompt needed), False if cancelled.
        """
        total = len(files)
        display_files = files[:max_display]
        
        logger.info(f"{title}: {total} files")
        for filepath in display_files:
            logger.info(f"  {filepath}")
        
        if total > max_display and show_all_prompt:
            if self.confirm(
                f"Showing {max_display} of {total}. Show all?",
                default=False,
                style="yellow"
            ):
                for filepath in files[max_display:]:
                    logger.info(f"  {filepath}")
        
        return True
    
    def show_warning(self, message: str):
        """
        Display a warning message.
        
        Args:
            message: Warning message to display.
        """
        self.console.print(f"[bold yellow]⚠️  {message}[/bold yellow]")
    
    def show_error(self, message: str):
        """
        Display an error message.
        
        Args:
            message: Error message to display.
        """
        self.console.print(f"[bold red]❌ {message}[/bold red]")
    
    def show_success(self, message: str):
        """
        Display a success message.
        
        Args:
            message: Success message to display.
        """
        self.console.print(f"[bold green]✅ {message}[/bold green]")
    
    def show_info(self, message: str):
        """
        Display an info message.
        
        Args:
            message: Info message to display.
        """
        self.console.print(f"[bold blue]ℹ️  {message}[/bold blue]")
    
    def create_table(
        self,
        title: str,
        columns: List[str],
        show_lines: bool = True
    ) -> Table:
        """
        Create a Rich table with consistent styling.
        
        Args:
            title: Table title.
            columns: List of column names.
            show_lines: Whether to show lines between rows.
            
        Returns:
            Configured Rich Table object.
        """
        table = Table(
            title=title,
            box=box.DOUBLE_EDGE,
            header_style="bold",
            border_style="blue",
            show_lines=show_lines,
        )
        
        for column in columns:
            table.add_column(column)
        
        return table
