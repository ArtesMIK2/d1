import sys
from PyQt5.QtWidgets import QApplication
from image_analyzer_app import ImageAnalyzerApp

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageAnalyzerApp()
    window.show()
    sys.exit(app.exec_())