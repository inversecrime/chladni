from simulation import Simulation
import sys
from typing import Callable
from PyQt6.QtWidgets import QApplication, QWidget, QSlider, QSpinBox, QVBoxLayout, QPushButton, QLabel, QDoubleSpinBox
from PyQt6 import QtCore
from PyQt6.QtCore import QTimer

from threading import Thread


class ControlWidget(QWidget):
    def __init__(self):
        super().__init__()

        mu_label = QLabel("mu:")

        mu_box = QDoubleSpinBox()
        mu_box.setRange(0, 1)
        mu_box.setValue(0.5)

        mu_button = QPushButton("Set mu")

        frequency_label = QLabel("Frequency:")

        frequency_box = QSpinBox()
        frequency_box.setRange(0, 100)
        frequency_box.setValue(0)

        frequency_slider = QSlider()
        frequency_slider.setOrientation(QtCore.Qt.Orientation.Horizontal)
        frequency_slider.setRange(0, 100)
        frequency_slider.setValue(0)

        particles_label = QLabel("Particles:")

        particles_box = QSpinBox()
        particles_box.setRange(0, 10000)
        particles_box.setValue(100)

        throw_particles_button = QPushButton("Throw Particles")

        keep_throwing_particles_button = QPushButton("Keep Throwing Particles")
        keep_throwing_particles_button.setCheckable(True)

        frequency_box.valueChanged.connect(lambda value: frequency_slider.setValue(value))
        frequency_slider.valueChanged.connect(lambda value: frequency_box.setValue(value))

        throw_particles_timer = QTimer(self)
        throw_particles_timer.timeout.connect(lambda: simulation.throw_particles(particles_box.value()))

        mu_button.clicked.connect(lambda: simulation.set_mu(mu_box.value()))
        frequency_box.valueChanged.connect(lambda value: simulation.set_frequency(value))
        throw_particles_button.clicked.connect(lambda: simulation.throw_particles(particles_box.value()))
        keep_throwing_particles_button.toggled.connect(lambda toggled: throw_particles_timer.start(500) if toggled else throw_particles_timer.stop())

        layout = QVBoxLayout()
        layout.addWidget(mu_label)
        layout.addWidget(mu_box)
        layout.addWidget(mu_button)
        layout.addWidget(frequency_label)
        layout.addWidget(frequency_box)
        layout.addWidget(frequency_slider)
        layout.addWidget(particles_label)
        layout.addWidget(particles_box)
        layout.addWidget(throw_particles_button)
        layout.addWidget(keep_throwing_particles_button)
        self.setLayout(layout)


app = QApplication(sys.argv)

window = ControlWidget()
window.show()

Thread(target=app.exec).start()

simulation = Simulation()
simulation.start()
