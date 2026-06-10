import sys
import math
import ast
import operator
import random

from PyQt6.QtCore import Qt, QTimer, QPointF
from PyQt6.QtGui import (
    QColor,
    QPainter,
    QPen,
    QBrush,
    QRadialGradient,
    QFont,
    QLinearGradient,
)
from PyQt6.QtWidgets import QApplication, QWidget


class SafeCalculator:
    """
    Safely evaluates calculator expressions.
    Supports numbers, +, -, *, /, %, **, and selected math functions.
    """

    allowed_operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
    }

    allowed_functions = {
        "sin": lambda x: math.sin(math.radians(x)),
        "cos": lambda x: math.cos(math.radians(x)),
        "tan": lambda x: math.tan(math.radians(x)),
        "sqrt": math.sqrt,
        "log": math.log10,
    }

    def evaluate(self, expression):
        expression = expression.replace("×", "*").replace("÷", "/").replace("^", "**")

        tree = ast.parse(expression, mode="eval")
        return self._eval_node(tree.body)

    def _eval_node(self, node):
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("Invalid value")

        if isinstance(node, ast.BinOp):
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            op_type = type(node.op)

            if op_type not in self.allowed_operators:
                raise ValueError("Invalid operator")

            return self.allowed_operators[op_type](left, right)

        if isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand)
            op_type = type(node.op)

            if op_type not in self.allowed_operators:
                raise ValueError("Invalid unary operator")

            return self.allowed_operators[op_type](operand)

        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Invalid function")

            func_name = node.func.id

            if func_name not in self.allowed_functions:
                raise ValueError("Function not allowed")

            if len(node.args) != 1:
                raise ValueError("Function needs one argument")

            arg = self._eval_node(node.args[0])
            return self.allowed_functions[func_name](arg)

        raise ValueError("Invalid expression")


class OrbitButton:
    def __init__(
        self,
        label,
        value,
        orbit_radius,
        angle,
        radius,
        color,
        kind="planet",
        speed=0.25,
    ):
        self.label = label
        self.value = value
        self.orbit_radius = orbit_radius
        self.angle = angle
        self.radius = radius
        self.color = color
        self.kind = kind
        self.speed = speed
        self.position = QPointF(0, 0)
        self.hovered = False

    def update_position(self, center):
        x = center.x() + math.cos(math.radians(self.angle)) * self.orbit_radius
        y = center.y() + math.sin(math.radians(self.angle)) * self.orbit_radius
        self.position = QPointF(x, y)

    def contains(self, point):
        dx = point.x() - self.position.x()
        dy = point.y() - self.position.y()
        return math.sqrt(dx * dx + dy * dy) <= self.radius + 5


class OrbitCalc(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("OrbitCalc - Solar System Calculator")
        self.setFixedSize(900, 720)
        self.setMouseTracking(True)

        self.expression = ""
        self.result = ""
        self.calculator = SafeCalculator()

        self.angle_offset = 0
        self.hover_button = None

        self.comet_active = False
        self.comet_angle = 0
        self.comet_life = 0

        self.stars = self.generate_stars(180)
        self.buttons = self.create_orbit_buttons()

        self.timer = QTimer()
        self.timer.timeout.connect(self.animate)
        self.timer.start(16)

    def generate_stars(self, count):
        stars = []
        for _ in range(count):
            stars.append(
                {
                    "x": random.randint(0, 900),
                    "y": random.randint(0, 720),
                    "size": random.choice([1, 1, 1, 2]),
                    "alpha": random.randint(80, 220),
                }
            )
        return stars

    def create_orbit_buttons(self):
        buttons = []

        number_color = QColor("#35f6ff")
        operator_color = QColor("#ff4df3")
        function_color = QColor("#b66cff")
        equal_color = QColor("#00ff88")
        black_hole_color = QColor("#090014")

        numbers = [
            ("7", 220, 220),
            ("8", 220, 280),
            ("9", 220, 340),
            ("4", 270, 190),
            ("5", 270, 250),
            ("6", 270, 310),
            ("1", 320, 210),
            ("2", 320, 270),
            ("3", 320, 330),
            ("0", 370, 270),
        ]

        for label, orbit, angle in numbers:
            buttons.append(
                OrbitButton(
                    label=label,
                    value=label,
                    orbit_radius=orbit,
                    angle=angle,
                    radius=28,
                    color=number_color,
                    kind="planet",
                    speed=0.05,
                )
            )

        operators = [
            ("+", "+", 230, 70),
            ("−", "-", 230, 120),
            ("×", "×", 230, 20),
            ("÷", "÷", 230, -35),
            ("%", "%", 280, -10),
            ("=", "=", 310, 70),
        ]

        for label, value, orbit, angle in operators:
            color = equal_color if label == "=" else operator_color
            buttons.append(
                OrbitButton(
                    label=label,
                    value=value,
                    orbit_radius=orbit,
                    angle=angle,
                    radius=26,
                    color=color,
                    kind="moon",
                    speed=-0.07,
                )
            )

        functions = [
            ("sin", "sin(", 360, 180),
            ("cos", "cos(", 360, 220),
            ("tan", "tan(", 360, 260),
            ("sqrt", "sqrt(", 360, 300),
            ("log", "log(", 360, 340),
            ("(", "(", 360, 20),
            (")", ")", 360, 60),
            (".", ".", 360, 100),
        ]

        for label, value, orbit, angle in functions:
            buttons.append(
                OrbitButton(
                    label=label,
                    value=value,
                    orbit_radius=orbit,
                    angle=angle,
                    radius=23,
                    color=function_color,
                    kind="asteroid",
                    speed=0.09,
                )
            )

        buttons.append(
            OrbitButton(
                label="●",
                value="clear",
                orbit_radius=150,
                angle=180,
                radius=31,
                color=black_hole_color,
                kind="blackhole",
                speed=-0.03,
            )
        )

        return buttons

    def animate(self):
        self.angle_offset += 0.4

        for button in self.buttons:
            button.angle += button.speed

        if self.comet_active:
            self.comet_angle += 8
            self.comet_life -= 1

            if self.comet_life <= 0:
                self.comet_active = False

        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        self.draw_background(painter)
        self.draw_stars(painter)

        center = QPointF(self.width() / 2, self.height() / 2)

        self.draw_orbits(painter, center)
        self.draw_asteroid_belt(painter, center)
        self.draw_sun_display(painter, center)

        for button in self.buttons:
            button.update_position(center)
            self.draw_orbit_button(painter, button)

        if self.comet_active:
            self.draw_comet(painter, center)

        self.draw_title(painter)
        self.draw_hint(painter)

    def draw_background(self, painter):
        gradient = QLinearGradient(0, 0, self.width(), self.height())
        gradient.setColorAt(0.0, QColor("#02030f"))
        gradient.setColorAt(0.45, QColor("#070b24"))
        gradient.setColorAt(1.0, QColor("#12001f"))

        painter.fillRect(self.rect(), gradient)

    def draw_stars(self, painter):
        for star in self.stars:
            color = QColor(255, 255, 255, star["alpha"])
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(color)
            painter.drawEllipse(
                QPointF(star["x"], star["y"]),
                star["size"],
                star["size"],
            )

    def draw_orbits(self, painter, center):
        orbit_radii = [150, 220, 270, 320, 360]

        for radius in orbit_radii:
            pen = QPen(QColor(80, 220, 255, 45), 1)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(center, radius, radius)

    def draw_asteroid_belt(self, painter, center):
        painter.setPen(Qt.PenStyle.NoPen)

        for i in range(80):
            angle = math.radians(i * 4.5 + self.angle_offset)
            radius = 340 + random.randint(-4, 4)
            x = center.x() + math.cos(angle) * radius
            y = center.y() + math.sin(angle) * radius

            painter.setBrush(QColor(180, 120, 255, 90))
            painter.drawEllipse(QPointF(x, y), 1.5, 1.5)

    def draw_sun_display(self, painter, center):
        sun_radius = 115

        glow = QRadialGradient(center, sun_radius * 1.8)
        glow.setColorAt(0.0, QColor(255, 214, 80, 255))
        glow.setColorAt(0.35, QColor(255, 120, 40, 180))
        glow.setColorAt(0.7, QColor(255, 60, 0, 65))
        glow.setColorAt(1.0, QColor(255, 60, 0, 0))

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(glow))
        painter.drawEllipse(center, sun_radius * 1.8, sun_radius * 1.8)

        body = QRadialGradient(center, sun_radius)
        body.setColorAt(0.0, QColor("#fff6a8"))
        body.setColorAt(0.45, QColor("#ffb000"))
        body.setColorAt(1.0, QColor("#ff5a00"))

        painter.setBrush(QBrush(body))
        painter.drawEllipse(center, sun_radius, sun_radius)

        painter.setPen(QPen(QColor("#ffffff"), 2))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(center, sun_radius, sun_radius)

        painter.setPen(QColor("#170000"))

        expression_font = QFont("Arial", 16, QFont.Weight.Bold)
        painter.setFont(expression_font)

        display_text = self.expression if self.expression else "OrbitCalc"

        if len(display_text) > 18:
            display_text = "..." + display_text[-18:]

        painter.drawText(
            int(center.x() - 90),
            int(center.y() - 30),
            180,
            40,
            Qt.AlignmentFlag.AlignCenter,
            display_text,
        )

        result_font = QFont("Arial", 22, QFont.Weight.Black)
        painter.setFont(result_font)

        result_text = self.result if self.result else "☀"

        if len(result_text) > 13:
            result_text = result_text[:13]

        painter.drawText(
            int(center.x() - 90),
            int(center.y() + 5),
            180,
            50,
            Qt.AlignmentFlag.AlignCenter,
            result_text,
        )

    def draw_orbit_button(self, painter, button):
        pos = button.position
        radius = button.radius

        if button.hovered:
            glow_radius = radius * 2.3
            glow_alpha = 180
        else:
            glow_radius = radius * 1.7
            glow_alpha = 90

        glow = QRadialGradient(pos, glow_radius)
        glow.setColorAt(0.0, QColor(button.color.red(), button.color.green(), button.color.blue(), glow_alpha))
        glow.setColorAt(0.7, QColor(button.color.red(), button.color.green(), button.color.blue(), 50))
        glow.setColorAt(1.0, QColor(button.color.red(), button.color.green(), button.color.blue(), 0))

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(glow))
        painter.drawEllipse(pos, glow_radius, glow_radius)

        if button.kind == "blackhole":
            self.draw_black_hole(painter, button)
            return

        planet = QRadialGradient(pos, radius)
        planet.setColorAt(0.0, QColor("#ffffff"))
        planet.setColorAt(0.35, button.color)
        planet.setColorAt(1.0, QColor("#071026"))

        painter.setBrush(QBrush(planet))
        painter.setPen(QPen(QColor(255, 255, 255, 160), 1.5))
        painter.drawEllipse(pos, radius, radius)

        if button.kind == "moon":
            painter.setPen(QPen(QColor(255, 255, 255, 80), 1))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(pos, radius + 6, radius + 6)

        if button.kind == "asteroid":
            painter.setPen(QPen(QColor(220, 180, 255, 150), 1))
            painter.drawLine(
                QPointF(pos.x() - radius * 0.8, pos.y() + radius * 0.3),
                QPointF(pos.x() + radius * 0.6, pos.y() - radius * 0.4),
            )

        painter.setPen(QColor("#ffffff"))

        if len(button.label) >= 3:
            font_size = 10
        else:
            font_size = 16

        painter.setFont(QFont("Arial", font_size, QFont.Weight.Bold))
        painter.drawText(
            int(pos.x() - radius),
            int(pos.y() - radius),
            int(radius * 2),
            int(radius * 2),
            Qt.AlignmentFlag.AlignCenter,
            button.label,
        )

    def draw_black_hole(self, painter, button):
        pos = button.position
        radius = button.radius

        outer = QRadialGradient(pos, radius * 2.2)
        outer.setColorAt(0.0, QColor("#000000"))
        outer.setColorAt(0.45, QColor("#220033"))
        outer.setColorAt(0.75, QColor("#8a00ff"))
        outer.setColorAt(1.0, QColor(0, 0, 0, 0))

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(outer))
        painter.drawEllipse(pos, radius * 2.1, radius * 2.1)

        painter.setBrush(QColor("#000000"))
        painter.setPen(QPen(QColor("#c77dff"), 2))
        painter.drawEllipse(pos, radius, radius)

        painter.setPen(QColor("#ffffff"))
        painter.setFont(QFont("Arial", 9, QFont.Weight.Bold))
        painter.drawText(
            int(pos.x() - radius),
            int(pos.y() - radius),
            int(radius * 2),
            int(radius * 2),
            Qt.AlignmentFlag.AlignCenter,
            "CLEAR",
        )

    def draw_comet(self, painter, center):
        angle = math.radians(self.comet_angle)
        radius = 260

        head_x = center.x() + math.cos(angle) * radius
        head_y = center.y() + math.sin(angle) * radius

        tail_x = center.x() + math.cos(angle - 0.6) * (radius - 90)
        tail_y = center.y() + math.sin(angle - 0.6) * (radius - 90)

        painter.setPen(QPen(QColor(0, 255, 180, 200), 5))
        painter.drawLine(QPointF(tail_x, tail_y), QPointF(head_x, head_y))

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#ffffff"))
        painter.drawEllipse(QPointF(head_x, head_y), 8, 8)

        painter.setBrush(QColor(0, 255, 180, 120))
        painter.drawEllipse(QPointF(head_x, head_y), 18, 18)

    def draw_title(self, painter):
        painter.setPen(QColor("#8dfcff"))
        painter.setFont(QFont("Arial", 28, QFont.Weight.Black))

        painter.drawText(
            0,
            20,
            self.width(),
            50,
            Qt.AlignmentFlag.AlignCenter,
            "ORBITCALC",
        )

        painter.setPen(QColor("#ff4df3"))
        painter.setFont(QFont("Arial", 11, QFont.Weight.Bold))

        painter.drawText(
            0,
            62,
            self.width(),
            30,
            Qt.AlignmentFlag.AlignCenter,
            "Navigate the galaxy to solve math",
        )

    def draw_hint(self, painter):
        painter.setPen(QColor(180, 220, 255, 150))
        painter.setFont(QFont("Arial", 10))

        painter.drawText(
            0,
            self.height() - 35,
            self.width(),
            25,
            Qt.AlignmentFlag.AlignCenter,
            "Tip: Use asteroid functions like sin(90), sqrt(25), log(100). Black hole clears everything.",
        )

    def mouseMoveEvent(self, event):
        mouse_pos = QPointF(event.position())

        self.hover_button = None

        for button in self.buttons:
            if button.contains(mouse_pos):
                button.hovered = True
                self.hover_button = button
                self.setCursor(Qt.CursorShape.PointingHandCursor)
            else:
                button.hovered = False

        if self.hover_button is None:
            self.setCursor(Qt.CursorShape.ArrowCursor)

        self.update()

    def mousePressEvent(self, event):
        mouse_pos = QPointF(event.position())

        for button in self.buttons:
            if button.contains(mouse_pos):
                self.handle_button_click(button.value)
                break

    def keyPressEvent(self, event):
        key = event.text()

        if key in "0123456789.+-*/()%":
            self.expression += key
            self.result = ""
        elif key == "\r" or key == "\n":
            self.calculate()
        elif event.key() == Qt.Key.Key_Backspace:
            self.expression = self.expression[:-1]
            self.result = ""
        elif event.key() == Qt.Key.Key_Escape:
            self.clear()

        self.update()

    def handle_button_click(self, value):
        if value == "clear":
            self.clear()
            return

        if value == "=":
            self.calculate()
            return

        self.expression += value
        self.result = ""
        self.update()

    def calculate(self):
        try:
            if not self.expression:
                return

            result = self.calculator.evaluate(self.expression)

            if isinstance(result, float):
                result = round(result, 8)

            self.result = str(result)
            self.expression = str(result)

            self.comet_active = True
            self.comet_angle = 0
            self.comet_life = 70

        except Exception:
            self.result = "ERROR"
            self.comet_active = True
            self.comet_angle = 180
            self.comet_life = 45

        self.update()

    def clear(self):
        self.expression = ""
        self.result = ""
        self.update()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = OrbitCalc()
    window.show()

    sys.exit(app.exec())