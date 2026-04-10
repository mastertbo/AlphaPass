"""AlphaPass application entry point."""

import sys

from loguru import logger


def main():
    """Launch the AlphaPass GUI application."""
    # Configure loguru
    logger.remove()
    logger.add(sys.stderr, level="INFO",
               format="<green>{time:HH:mm:ss}</green> | "
                      "<level>{level: <8}</level> | "
                      "<cyan>{message}</cyan>")

    logger.info("Starting AlphaPass...")

    # Ensure correct PyTorch variant before importing torch-dependent modules
    from AlphaPass.utils.bootstrap import ensure_correct_torch
    ensure_correct_torch()

    from PySide6.QtWidgets import QApplication
    from AlphaPass.ui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("AlphaPass")
    app.setStyle("Fusion")

    # Theme is applied by MainWindow.__init__ via _apply_theme()
    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
