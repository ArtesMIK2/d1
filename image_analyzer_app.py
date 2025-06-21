import sys
import cv2
import numpy as np
import pydicom
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QFileDialog, QVBoxLayout,
    QWidget, QSpinBox, QHBoxLayout, QMessageBox, QInputDialog
)
from PyQt5.QtGui import QImage, QPixmap, QCursor
from PyQt5.QtCore import Qt, QPoint
import os
import logging
import time

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class ImageAnalyzerApp(QMainWindow):
    """Класс приложения для анализа DICOM-серий с улучшенным выделением и масштабированием."""
    
    def __init__(self):
        super().__init__()
        logging.info("Инициализация приложения")
        self.setWindowTitle("Анализатор DICOM Изображений")
        self.setGeometry(100, 100, 800, 600)

        # Интерфейс
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        # Отображение изображения
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMouseTracking(True)
        self.image_label.setMinimumSize(400, 400)
        self.image_label.setFocusPolicy(Qt.StrongFocus)
        self.main_layout.addWidget(self.image_label)

        # Контейнер для кнопок
        self.button_container = QWidget()
        self.button_layout = QHBoxLayout(self.button_container)
        self.main_layout.addWidget(self.button_container)

        # Кнопки
        self.load_button = QPushButton("Загрузить DICOM серию")
        self.load_button.clicked.connect(self.load_image)
        self.button_layout.addWidget(self.load_button)

        self.add_region_button = QPushButton("Добавить область")
        self.add_region_button.setCheckable(True)
        self.add_region_button.clicked.connect(self.toggle_add_region)
        self.button_layout.addWidget(self.add_region_button)

        self.remove_region_button = QPushButton("Удалить область")
        self.remove_region_button.clicked.connect(self.remove_last_region)
        self.button_layout.addWidget(self.remove_region_button)

        self.analyze_button = QPushButton("Анализировать")
        self.analyze_button.clicked.connect(self.analyze_image)
        self.button_layout.addWidget(self.analyze_button)

        self.show_binary_button = QPushButton("Показать бинарное")
        self.show_binary_button.clicked.connect(self.show_binary)
        self.button_layout.addWidget(self.show_binary_button)

        self.area_info_button = QPushButton("Площадь области")
        self.area_info_button.clicked.connect(self.show_area_of_region)
        self.button_layout.addWidget(self.area_info_button)

        self.restart_button = QPushButton("Рестарт")
        self.restart_button.clicked.connect(self.restart)
        self.button_layout.addWidget(self.restart_button)

        # Навигация по срезам
        slice_layout = QHBoxLayout()
        prev_button = QPushButton("Пред. срез")
        next_button = QPushButton("След. срез")
        prev_button.clicked.connect(self.prev_slice)
        next_button.clicked.connect(self.next_slice)
        slice_layout.addWidget(prev_button)
        slice_layout.addWidget(next_button)
        self.main_layout.addLayout(slice_layout)

        # Параметры анализа
        self.param_widgets = []
        self.create_param_controls()

        # Результаты
        self.result_label = QLabel("Результаты будут здесь")
        self.main_layout.addWidget(self.result_label)

        # Переменные
        self.image = None
        self.regions = []
        self.current_slice = 0
        self.drawing = False
        self.ix, self.iy = -1, -1
        self.selected_region = None
        self.handle = None
        self.zoom_level = 1.0
        self.pan_offset = QPoint(0, 0)
        self.panning = False
        self.last_mouse_pos = None
        self.add_region_mode = False
        self.pixel_spacing = None  # Новый атрибут для размера пикселя

    def create_param_controls(self):
        """Создание элементов управления параметрами анализа."""
        logging.info("Создание элементов управления")
        params = [
            ("Минимальная площадь:", 150, 10, 1000),
            ("Мин. соотношение сторон:", 2, 1, 10),
            ("Макс. соотношение сторон:", 50, 10, 100)
        ]
        for label_text, default, min_val, max_val in params:
            layout = QHBoxLayout()
            label = QLabel(label_text)
            spin = QSpinBox()
            spin.setRange(min_val, max_val)
            spin.setValue(default)
            layout.addWidget(label)
            layout.addWidget(spin)
            self.main_layout.addLayout(layout)
            self.param_widgets.append(spin)
        self.area_spin, self.min_ratio_spin, self.max_ratio_spin = self.param_widgets

    def toggle_add_region(self):
        """Переключение режима добавления области."""
        self.add_region_mode = self.add_region_button.isChecked()
        logging.info(f"Режим добавления области: {'включен' if self.add_region_mode else 'выключен'}")
        if self.add_region_mode:
            self.add_region_button.setStyleSheet("background-color: lightgreen")
        else:
            self.add_region_button.setStyleSheet("")

    def load_image(self):
        """Загрузка DICOM-серии из папки."""
        logging.info("Начало загрузки DICOM-серии")
        start_time = time.time()
        folder = QFileDialog.getExistingDirectory(self, "Выберите папку с DICOM")
        if not folder:
            self.result_label.setText("Папка не выбрана")
            logging.info("Папка не выбрана")
            return
        try:
            dicom_files = [
                os.path.join(folder, f) for f in os.listdir(folder)
                if f.lower().endswith((".dcm", ".dicom"))
            ]
            if not dicom_files:
                self.result_label.setText("Нет DICOM файлов в папке")
                logging.error("Нет DICOM файлов в папке")
                return
            slices = []
            self.pixel_spacing = None
            for f in sorted(dicom_files):
                logging.debug(f"Чтение файла: {f}")
                try:
                    ds = pydicom.dcmread(f)
                    if hasattr(ds, "pixel_array"):
                        slices.append(ds)
                        if self.pixel_spacing is None and hasattr(ds, "PixelSpacing"):
                            self.pixel_spacing = ds.PixelSpacing
                            logging.info(f"Pixel Spacing: {self.pixel_spacing} мм")
                except Exception as e:
                    logging.error(f"Ошибка чтения файла {f}: {e}")
            if not slices:
                self.result_label.setText("Нет изображений в DICOM файлах")
                logging.error("Нет изображений в DICOM файлах")
                return
            slices.sort(key=lambda x: x.get("InstanceNumber", 0))
            self.image = np.stack([s.pixel_array for s in slices], axis=0)
            if np.max(self.image) > 0:
                self.image = (self.image / np.max(self.image) * 255).astype(np.uint8)
            else:
                logging.warning("Максимальное значение изображения равно 0")
                self.image = self.image.astype(np.uint8)
            self.regions = []
            self.current_slice = 0
            self.zoom_level = 1.0
            self.pan_offset = QPoint(0, 0)
            self.add_region_mode = False
            self.add_region_button.setChecked(False)
            self.add_region_button.setStyleSheet("")
            self.image_label.setFocus()
            self.display_slice()
            if self.pixel_spacing is None:
                logging.warning("Pixel Spacing не найден в DICOM-файлах")
            logging.info(f"Загружено {len(slices)} срезов за {time.time() - start_time:.2f} сек")
        except Exception as e:
            self.result_label.setText(f"Ошибка загрузки: {e}")
            logging.error(f"Ошибка загрузки: {e}")

    def display_slice(self):
        """Отображение текущего среза с выделенными областями."""
        logging.debug(f"Отображение среза {self.current_slice}")
        if self.image is None:
            logging.warning("Изображение не загружено")
            return
        slice_image = self.image[self.current_slice]
        if len(slice_image.shape) == 2:
            slice_image = cv2.cvtColor(slice_image, cv2.COLOR_GRAY2BGR)
        img_copy = slice_image.copy()
        alpha = 0.3
        overlay = img_copy.copy()
        for region in [r for r in self.regions if r[0] == self.current_slice]:
            _, x1, y1, x2, y2 = region
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            if x1 < x2 and y1 < y2:
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), thickness=cv2.FILLED)
        img_with_overlay = cv2.addWeighted(img_copy, 1 - alpha, overlay, alpha, 0)
        for region in [r for r in self.regions if r[0] == self.current_slice]:
            _, x1, y1, x2, y2 = region
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            if x1 < x2 and y1 < y2:
                cv2.rectangle(img_with_overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
                if self.selected_region is not None and region == self.regions[self.selected_region]:
                    cv2.rectangle(img_with_overlay, (x1-3, y1-3), (x1+3, y1+3), (255, 0, 0), -1)  # TL
                    cv2.rectangle(img_with_overlay, (x2-3, y1-3), (x2+3, y1+3), (255, 0, 0), -1)  # TR
                    cv2.rectangle(img_with_overlay, (x1-3, y2-3), (x1+3, y2+3), (255, 0, 0), -1)  # BL
                    cv2.rectangle(img_with_overlay, (x2-3, y2-3), (x2+3, y2+3), (255, 0, 0), -1)  # BR
        self.display_image(img_with_overlay)

    def display_image(self, image):
        """Отображение изображения с учетом масштаба и панорамирования."""
        logging.debug("Отображение изображения")
        if image is None or image.size == 0:
            self.result_label.setText("Ошибка: изображение пустое")
            logging.error("Изображение пустое")
            return
        height, width, channel = image.shape
        zoomed_width = max(1, int(width * self.zoom_level))
        zoomed_height = max(1, int(height * self.zoom_level))
        try:
            zoomed_image = cv2.resize(image, (zoomed_width, zoomed_height), interpolation=cv2.INTER_LINEAR)
            canvas_width = self.image_label.width()
            canvas_height = self.image_label.height()
            canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
            x_offset = int(self.pan_offset.x() + canvas_width // 2 - zoomed_width // 2)
            y_offset = int(self.pan_offset.y() + canvas_height // 2 - zoomed_height // 2)
            x_start_canvas = max(0, x_offset)
            y_start_canvas = max(0, y_offset)
            x_end_canvas = min(canvas_width, x_offset + zoomed_width)
            y_end_canvas = min(canvas_height, y_offset + zoomed_height)
            x_start_image = max(0, -x_offset)
            y_start_image = max(0, -y_offset)
            x_end_image = x_start_image + (x_end_canvas - x_start_canvas)
            y_end_image = y_start_image + (y_end_canvas - y_start_canvas)
            if x_end_canvas > x_start_canvas and y_end_canvas > y_start_canvas:
                canvas[y_start_canvas:y_end_canvas, x_start_canvas:x_end_canvas] = \
                    zoomed_image[y_start_image:y_end_image, x_start_image:x_end_image]
            bytes_per_line = 3 * canvas_width
            q_image = QImage(canvas.data, canvas_width, canvas_height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            self.image_label.setPixmap(QPixmap.fromImage(q_image))
            logging.debug("Изображение отображено")
        except Exception as e:
            self.result_label.setText(f"Ошибка отображения: {e}")
            logging.error(f"Ошибка отображения: {e}")

    def start_new_region(self):
        """Начало выделения новой области."""
        if self.image is not None and self.add_region_mode:
            self.drawing = True
            self.selected_region = None
            self.handle = None
            self.ix, self.iy = -1, -1
            logging.info("Начало выделения области")

    def mousePressEvent(self, event):
        """Обработка нажатия мыши."""
        logging.debug(f"Нажатие мыши: кнопка {event.button()}")
        if self.image is None:
            return
        x, y = self.transform_mouse_pos(event.x(), event.y())
        if event.button() == Qt.LeftButton:
            cursor_set = False
            for i, region in enumerate([r for r in self.regions if r[0] == self.current_slice]):
                _, x1, y1, x2, y2 = region
                handles = {
                    "tl": (x1, y1), "tr": (x2, y1),
                    "bl": (x1, y2), "br": (x2, y2)
                }
                for h, (hx, hy) in handles.items():
                    if abs(x - hx) < 10 / self.zoom_level and abs(y - hy) < 10 / self.zoom_level:
                        self.image_label.setCursor(QCursor(Qt.SizeAllCursor))
                        self.selected_region = self.regions.index(region)
                        self.handle = h
                        self.ix, self.iy = x, y
                        self.drawing = False
                        logging.debug(f"Выбран угол {h} области {self.selected_region}")
                        return
                if x1 <= x <= x2 and y1 <= y <= y2:
                    self.image_label.setCursor(QCursor(Qt.SizeAllCursor))
                    self.selected_region = self.regions.index(region)
                    self.handle = "move"
                    self.ix, self.iy = x, y
                    self.drawing = False
                    logging.debug(f"Выбрана область {self.selected_region} для перемещения")
                    return
            if not cursor_set:
                self.image_label.setCursor(QCursor(Qt.ArrowCursor))
            if self.add_region_mode:
                self.drawing = True
                self.selected_region = None
                self.handle = None
                self.ix, self.iy = x, y
                self.regions.append((self.current_slice, self.ix, self.iy, self.ix, self.iy))
                logging.debug(f"Начало новой области в ({x},{y})")
        elif event.button() == Qt.MidButton:
            self.panning = True
            self.last_mouse_pos = QPoint(event.x(), event.y())
            logging.debug("Начало панорамирования")

    def mouseMoveEvent(self, event):
        """Обработка движения мыши."""
        if self.image is None:
            return
        x, y = self.transform_mouse_pos(event.x(), event.y())
        cursor_set = False
        for region in [r for r in self.regions if r[0] == self.current_slice]:
            _, x1, y1, x2, y2 = region
            handles = {
                "tl": (x1, y1), "tr": (x2, y1),
                "bl": (x1, y2), "br": (x2, y2)
            }
            for h, (hx, hy) in handles.items():
                if abs(x - hx) < 10 / self.zoom_level and abs(y - hy) < 10 / self.zoom_level:
                    self.image_label.setCursor(QCursor(Qt.SizeAllCursor))
                    cursor_set = True
                    break
            if x1 <= x <= x2 and y1 <= y <= y2:
                self.image_label.setCursor(QCursor(Qt.SizeAllCursor))
                cursor_set = True
                break
        if not cursor_set:
            self.image_label.setCursor(QCursor(Qt.ArrowCursor))
        if self.drawing and self.add_region_mode:
            self.regions[-1] = (self.current_slice, self.ix, self.iy, x, y)
            self.display_slice()
            logging.debug(f"Рисование области: ({x},{y})")
        elif self.handle:
            idx = self.selected_region
            slice_idx, x1, y1, x2, y2 = self.regions[idx]
            if self.handle == "tl":
                x1, y1 = x, y
            elif self.handle == "tr":
                x2, y1 = x, y
            elif self.handle == "bl":
                x1, y2 = x, y
            elif self.handle == "br":
                x2, y2 = x, y
            elif self.handle == "move":
                dx, dy = x - self.ix, y - self.iy
                x1 += dx
                x2 += dx
                y1 += dy
                y2 += dy
                self.ix, self.iy = x, y
            x1, x2 = sorted([x1, x2])
            y1, y2 = sorted([y1, y2])
            h, w = self.image.shape[1:3]
            x1, x2 = max(0, x1), min(w - 1, x2)
            y1, y2 = max(0, y1), min(h - 1, y2)
            self.regions[idx] = (slice_idx, x1, y1, x2, y2)
            self.display_slice()
            logging.debug(f"Обновление области {idx}: ({x1},{y1})-({x2},{y2})")
        elif self.panning:
            current_pos = QPoint(event.x(), event.y())
            delta = current_pos - self.last_mouse_pos
            self.pan_offset += delta
            self.last_mouse_pos = current_pos
            self.display_slice()
            logging.debug(f"Панорамирование: смещение {delta}")

    def mouseReleaseEvent(self, event):
        """Обработка отпускания мыши."""
        logging.debug(f"Отпускание мыши: {event.button()}")
        if self.image is None:
            return
        if event.button() == Qt.LeftButton and self.drawing and self.add_region_mode:
            self.drawing = False
            x, y = self.transform_mouse_pos(event.x(), event.y())
            x1, y1 = self.ix, self.iy
            x1, x2 = sorted([x1, x])
            y1, y2 = sorted([y1, y])
            h, w = self.image.shape[1:3]
            x1, x2 = max(0, x1), min(w - 1, x2)
            y1, y2 = max(0, y1), min(h - 1, y2)
            if x2 - x1 > 2 and y2 - y1 > 2:
                self.regions[-1] = (self.current_slice, x1, y1, x2, y2)
                self.selected_region = len(self.regions) - 1
                logging.info(f"Создана область: ({x1},{y1})-({x2},{y2})")
            else:
                self.regions.pop()
                logging.debug("Область слишком мала, удалена")
            self.display_slice()
        elif event.button() == Qt.LeftButton and self.handle:
            self.handle = None
            logging.debug("Завершено редактирование области")
        elif event.button() == Qt.MidButton:
            self.panning = False
            self.last_mouse_pos = None
            logging.debug("Панорамирование завершено")

    def wheelEvent(self, event):
        """Обработка колеса мыши для масштабирования."""
        if self.image is None:
            return
        logging.debug("Обработка масштабирования")
        delta = event.angleDelta().y()
        zoom_factor = 1.1 if delta > 0 else 1 / 1.1
        old_zoom = self.zoom_level
        self.zoom_level = max(0.5, min(self.zoom_level * zoom_factor, 10.0))
        mouse_x, mouse_y = event.x(), event.y()
        image_x, image_y = self.transform_mouse_pos(mouse_x, mouse_y)
        new_x = int(self.pan_offset.x() + (image_x * old_zoom - image_x * self.zoom_level))
        new_y = int(self.pan_offset.y() + (image_y * old_zoom - image_y * self.zoom_level))
        self.pan_offset.setX(new_x)
        self.pan_offset.setY(new_y)
        self.display_slice()
        logging.info(f"Масштаб изменен: {self.zoom_level:.2f}x")

    def keyPressEvent(self, event):
        """Обработка нажатий клавиш для перемещения."""
        if self.image is None:
            return
        step = int(20 / self.zoom_level)
        if event.key() == Qt.Key_Up:
            self.pan_offset.setY(self.pan_offset.y() + step)
            logging.info("Перемещение вверх")
        elif event.key() == Qt.Key_Down:
            self.pan_offset.setY(self.pan_offset.y() - step)
            logging.info("Перемещение вниз")
        elif event.key() == Qt.Key_Left:
            self.pan_offset.setX(self.pan_offset.x() + step)
            logging.info("Перемещение влево")
        elif event.key() == Qt.Key_Right:
            self.pan_offset.setX(self.pan_offset.x() - step)
            logging.info("Перемещение вправо")
        self.display_slice()

    def transform_mouse_pos(self, x, y):
        """Преобразование координат мыши в координаты изображения."""
        if self.image is None:
            return x, y
        width, height = self.image.shape[2], self.image.shape[1]
        zoomed_width = width * self.zoom_level
        zoomed_height = height * self.zoom_level
        x_offset = self.pan_offset.x() + self.image_label.width() // 2 - zoomed_width // 2
        y_offset = self.pan_offset.y() + self.image_label.height() // 2 - zoomed_height // 2
        image_x = (x - x_offset) / self.zoom_level
        image_y = (y - y_offset) / self.zoom_level
        return image_x, image_y

    def remove_last_region(self):
        """Удаление последней выделенной области."""
        logging.info("Удаление последней области")
        if self.regions:
            self.regions.pop()
            self.selected_region = None
            self.display_slice()

    def prev_slice(self):
        """Переход к предыдущему срезу."""
        logging.info("Переход к предыдущему срезу")
        if self.image is not None and self.current_slice > 0:
            self.current_slice -= 1
            self.selected_region = None
            self.display_slice()

    def next_slice(self):
        """Переход к следующему срезу."""
        logging.info("Переход к следующему срезу")
        if self.image is not None and self.current_slice < self.image.shape[0] - 1:
            self.current_slice += 1
            self.selected_region = None
            self.display_slice()

    def show_binary(self):
        """Отображение бинарного изображения для текущей области."""
        logging.info("Отображение бинарного изображения")
        start_time = time.time()
        if not self.regions or self.image is None:
            self.result_label.setText("Нет областей или изображения")
            logging.warning("Нет областей или изображения")
            return
        region = [r for r in self.regions if r[0] == self.current_slice][0]
        _, x1, y1, x2, y2 = region
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        roi = self.image[self.current_slice, y1:y2, x1:x2]
        if roi.size == 0:
            self.result_label.setText("Ошибка: область пустая")
            logging.error("Область пустая")
            return
        if len(roi.shape) == 2:
            gray = roi
        else:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        binary_color = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
        self.display_image(binary_color)
        logging.info(f"Бинарное изображение отображено за {time.time() - start_time:.3f} сек")

    def analyze_image(self):
        """Анализ изображения в текущем срезе."""
        logging.info("Начало анализа изображения")
        start_time = time.time()
        if self.image is None or not self.regions:
            self.result_label.setText("Нет изображения или регионов")
            logging.warning("Нет изображения или регионов")
            return
        total_objects = 0
        slice_image = self.image[self.current_slice]
        if len(slice_image.shape) == 2:
            slice_image = cv2.cvtColor(slice_image, cv2.COLOR_GRAY2BGR)
        alpha = 0.3
        overlay = slice_image.copy()
        min_contour_area = self.area_spin.value()
        min_aspect_ratio = self.min_ratio_spin.value() / 10
        max_aspect_ratio = self.max_ratio_spin.value() / 10
        for region in [r for r in self.regions if r[0] == self.current_slice]:
            _, x1, y1, x2, y2 = region
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            roi = slice_image[y1:y2, x1:x2]
            if roi.size == 0:
                logging.warning(f"Пустая область: ({x1},{y1})-({x2},{y2})")
                continue
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid_contours = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = w / h if h > 0 else 0
                if area > min_contour_area and min_aspect_ratio < aspect_ratio < max_aspect_ratio:
                    valid_contours.append(cnt)
            cv2.drawContours(overlay[y1:y2, x1:x2], valid_contours, -1, (0, 0, 255), 2)
            total_objects += len(valid_contours)
        img_with_overlay = cv2.addWeighted(slice_image, 1 - alpha, overlay, alpha, 0)
        self.result_label.setText(f"Всего объектов: {total_objects}")
        self.display_image(img_with_overlay)
        logging.info(f"Анализ завершен: {total_objects} объектов за {time.time() - start_time:.3f} сек")

    def show_area_of_region(self):
        """Отображение площади выбранной области."""
        logging.info("Запрос площади области")
        if not self.regions:
            QMessageBox.warning(self, "Ошибка", "Нет выделенных областей.")
            logging.warning("Нет областей")
            return
        index, ok = QInputDialog.getInt(self, "Введите номер области",
                                        f"Введите номер области (от 1 до {len(self.regions)}):",
                                        min=1, max=len(self.regions), step=1)
        if ok:
            region = self.regions[index - 1]
            _, x1, y1, x2, y2 = region
            width = abs(x2 - x1)
            height = abs(y2 - y1)
            area_pixels = width * height
            message = f"Площадь области #{index}:\n{area_pixels} пикселей²"
            if self.pixel_spacing:
                area_mm2 = area_pixels * self.pixel_spacing[0] * self.pixel_spacing[1]
                message += f"\n{area_mm2:.2f} мм²"
            else:
                message += "\nФизическая площадь недоступна (Pixel Spacing не найден)"
            QMessageBox.information(self, f"Площадь области #{index}", message)
            logging.info(f"Площадь области #{index}: {area_pixels} пикселей², "
                         f"{'{area_mm2:.2f} мм²' if self.pixel_spacing else 'без физической площади'}")

    def restart(self):
        """Сброс всех данных."""
        logging.info("Рестарт приложения")
        self.image = None
        self.regions = []
        self.current_slice = 0
        self.drawing = False
        self.ix, self.iy = -1, -1
        self.selected_region = None
        self.handle = None
        self.zoom_level = 1.0
        self.pan_offset = QPoint(0, 0)
        self.panning = False
        self.last_mouse_pos = None
        self.add_region_mode = False
        self.pixel_spacing = None
        self.add_region_button.setChecked(False)
        self.add_region_button.setStyleSheet("")
        self.image_label.clear()
        self.result_label.setText("Результаты будут здесь")
        self.display_slice()

if __name__ == "__main__":
    logging.info("Запуск приложения")
    app = QApplication(sys.argv)
    window = ImageAnalyzerApp()
    window.show()
    sys.exit(app.exec_())