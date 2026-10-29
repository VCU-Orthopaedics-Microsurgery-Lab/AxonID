"""
utils/logger.py: console output and optional plain-text log file for every entry point.

Usage
    from utils.logger import AxonIDLogger
    log = AxonIDLogger(log_path=path_or_None, program="AxonID Segment")
    log.info(...); log.warn(...); log.rule("SECTION"); log.finalize(summary={...})

Outputs
    Timestamped messages on the console; when log_path is given, the same
    messages plus an environment header (versions, GPU) in a text file.

Depends on
    rich, utils/version.py
"""

# ── Standard library ──────────────────────────────────────────────────────────
import os
import re
from datetime import datetime
 
# ── Third-party ───────────────────────────────────────────────────────────────
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

# ── Local ─────────────────────────────────────────────────────────────────────
from utils.version import __version__, get_env_info


class AxonIDLogger:
    def __init__(
        self,
        log_path: str = None,
        program:  str = "AxonID",
        context:  dict = None
    ):
        """
        Args:
            log_path: Path to log file. If None logging is disabled.
            program:  Program name shown in log header.
            context:  Optional dict of key-value pairs written into header
                      e.g. {'Model': 'rabbit_40x_v1.keras', 'Magnification': '40X'}
        """
        self.console    = Console()
        self.log_path   = log_path
        self.program    = program
        self.context    = context or {}
        self._t_start   = datetime.now()

        if log_path:
            # A bare filename with no directory would make dirname("")
            # raise in os.makedirs — but abspath() runs first here, so
            # dirname() always returns at least a drive/cwd root, never
            # empty. No caller contract is required.
            os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
            self._write_log_header()


    def _ts(self) -> str:
        """Current time as HH:MM:SS for per-line timestamps."""
        return datetime.now().strftime('%H:%M:%S')


    def _write_log_header(self):
        """Write structured header with run context."""
        try:
            # Deliberately local and try/except-guarded, not top-level:
            # AxonIDLogger is imported by modules with no other reason to
            # depend on torch. A module-level import would make logging
            # itself fail wherever torch isn't installed, instead of
            # falling back to 'unknown' here.
            import torch
            
            if torch.cuda.is_available():
                gpu = ', '.join(
                    torch.cuda.get_device_name(i)
                    for i in range(torch.cuda.device_count())
                )
            else:
                gpu = 'CPU'
        except Exception:
            gpu = 'unknown'

        env = get_env_info()
        lines = [
            f"{'=' * 72}",
            f"{self.program} LOG",
            f"AxonID v{__version__}",
            f"Python: {env['python']} | PyTorch: {env['pytorch']} | CUDA: {env['cuda']}",
            f"GPU        : {gpu}",
            f"{'=' * 72}",
            f"Start time : {self._t_start.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Log file   : {self.log_path}",
        ]
        if self.context:
            for k, v in self.context.items():
                lines.append(f"{k:<11}: {v}")
        lines.append(f"{'=' * 72}")
        lines.append("")

        with open(self.log_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')
            
            
    def _append(self, text: str, tag: str = ""):
        """
        Append timestamped plain text to log file.
        Format: [HH:MM:SS] [TAG]  message
        """
        if self.log_path:
            ts      = self._ts()
            prefix  = f"[{ts}] {tag}" if tag else f"[{ts}]"
            with open(self.log_path, 'a', encoding='utf-8') as f:
                f.write(f"{prefix} {text}\n")


    def _append_raw(self, text: str):
        """Append text with no timestamp — for headers, dividers, sections."""
        if self.log_path:
            with open(self.log_path, 'a', encoding='utf-8') as f:
                f.write(text + '\n')


    def info(self, msg: str):
        self.console.print(f"[cyan]ℹ[/cyan]  {msg}")
        self._append(f"[INFO]  {msg}")


    def success(self, msg: str):
        self.console.print(f"[green]✔[/green]  {msg}")
        self._append(f"[OK]    {msg}")


    def warn(self, msg: str):
        self.console.print(f"[yellow]⚠[/yellow]  {msg}")
        self._append(f"[WARN]  {msg}")


    def error(self, msg: str):
        self.console.print(f"[red]✖[/red]  {msg}")
        self._append(f"[ERROR] {msg}")


    def config_line(self, label: str, value: str, source: str = "", level: str = "info"):
        """
        Log one config-summary line with a fixed-width label, a value that
        wraps with a hanging indent if it's long, and an optional source
        tag printed dimmed on its own trailing position — instead of the
        label/value/source all being one long unwrapped string.

        Used by the format_*_config_summary() banners in utils/config.py;
        not intended for ordinary one-line info/warn/error messages.
        """
        icon, style = {
            "info": ("ℹ", "cyan"),
            "warn": ("⚠", "yellow"),
        }[level]
        label_str = f"{label:<26}"
        source_str = f"  [dim]{source}[/dim]" if source else ""
        self.console.print(
            f"[{style}]{icon}[/{style}]  {label_str}{value}{source_str}",
            soft_wrap=False,
        )
        plain_source = f"  {source}" if source else ""
        self._append(
            f"[{level.upper():<5}] {label_str}{value}{plain_source}"
        )


    def rule(self, title: str = ""):
        """Console rule + log section divider."""
        self.console.print()
        self.console.rule(f"[bold]{title}[/bold]")
        divider = f"\n── {title} {'─' * max(0, 68 - len(title))}"
        self._append_raw(divider)


    def print(self, msg: str):
        """Raw print — Rich markup on console, plain text in log."""
        self.console.print(msg)
        plain = re.sub(r'\[.*?\]', '', msg)
        self._append(plain)


    def log_dict(self, data: dict, title: str = ""):
        """Log a dictionary as aligned key: value pairs."""
        if title:
            self._append_raw(f"\n{title}")
        max_key = max(len(k) for k in data.keys()) if data else 0
        for k, v in data.items():
            line = f"  {k:<{max_key}} : {v}"
            self.console.print(line)
            self._append_raw(f"  {k:<{max_key}} : {v}")


    def write_section(self, title: str, content: str):
        """
        Write a named section directly to log only — not console.
        Used for large structured data like epoch tables and JSON metadata.
        Parseable by grep: search for '=== TITLE ===' to extract sections.
        """
        if self.log_path:
            with open(self.log_path, 'a', encoding='utf-8') as f:
                f.write(
                    f"\n{'=' * 72}\n"
                    f"=== {title} ===\n"
                    f"{'=' * 72}\n"
                    f"{content}\n"
                    f"{'=' * 72}\n"
                )


    def finalize(
        self,
        summary: dict = None,
        title: str = "RUN COMPLETE",
        elapsed_label: str = "Elapsed",
    ):
        """
        Write structured footer to log and console.
        Includes end time (log only, unchanged) and elapsed duration —
        now actually printed, to both console and log, instead of being
        computed and discarded. elapsed_label lets a caller with a more
        specific duration concept (e.g. "Total time (call to end)")
        override the generic default, rather than computing and adding
        its own separately-labeled duration into summary.
        """
        t_end     = datetime.now()
        elapsed   = t_end - self._t_start
        mins      = int(elapsed.total_seconds() // 60)
        secs      = int(elapsed.total_seconds() % 60)
        elapsed_str = f"{mins}m {secs}s"

        self.rule(title)

        self.console.print(f"  {elapsed_label} : {elapsed_str}")
        
        footer_lines = [
            f"End time : {t_end.strftime('%Y-%m-%d %H:%M:%S')}",
            f"{elapsed_label} : {elapsed_str}",
        ]
        if summary:
            footer_lines.append("")
            max_key = max(len(k) for k in summary.keys()) if summary else 0
            for k, v in summary.items():
                footer_lines.append(f"  {k:<{max_key}} : {v}")
                self.console.print(f"  {k:<{max_key}} : {v}")

        footer_lines.append(f"\n{'=' * 72}")

        self._append_raw('\n'.join(footer_lines))


    def progress(self):
        """Return a Rich Progress context manager pre-configured for AxonID."""
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=self.console
        )