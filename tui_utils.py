import shutil
from colorama import Fore, Style

def term_width(default: int = 80) -> int:
    try:
        return shutil.get_terminal_size().columns
    except Exception:
        return default

def center_text(text: str, width: int | None = None) -> str:
    if width is None:
        width = term_width()
    if len(text) >= width:
        return text
    return text.center(width)

def print_header(text: str,
                 color: str = Fore.GREEN, 
                 bright: bool = True,
                 underline: bool = False) -> None:
    """Centers a header line in the terminal. Styles wrap the centered line so ANSI codes dont affect centering."""
    w = term_width()
    line = center_text(text, w) 
    prefix = color + (Style.BRIGHT if bright else "")
    print(prefix + line + Style.RESET_ALL)
    if underline:
        ul = center_text("\u2500" * len(text), w) 
        print(prefix + ul + Style.RESET_ALL)

def print_center(text: str, color: str = Fore.GREEN, bright: bool = False) -> None:
    w = term_width()
    line = center_text(text, w)
    prefix = color + (Style.BRIGHT if bright else "")
    print(prefix + line + Style.RESET_ALL)
