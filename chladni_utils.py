import sys
from math import sqrt
from typing import Any, Literal

import numpy as np
import pygame
from matplotlib import pyplot as plt
from pygame.time import Clock
from PyQt6 import QtCore
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QImage, QPainter
from PyQt6.QtWidgets import (QApplication, QDoubleSpinBox, QLabel, QPushButton,
                             QSlider, QSpinBox, QVBoxLayout, QWidget)

import chladni_maths
from chladni_simulation import Simulation


class SimulationWindow(QWidget):
    simulation: Simulation
    clock: Clock
    image: QImage

    def __init__(self, simulation: Simulation) -> None:
        super().__init__()

        self.simulation = simulation
        self.clock = Clock()

        timer = QTimer(self)
        timer.timeout.connect(lambda: self.simulation_step())
        timer.start(16)

        timer = QTimer(self)
        timer.timeout.connect(lambda: self.simulation.update_particle_speeds())
        timer.start(200)

    def simulation_step(self) -> None:
        dt = self.clock.tick(60)
        self.simulation.step(dt)

        canvas = pygame.Surface((self.width(), self.height()))
        self.simulation.draw(canvas)
        self.image = QImage(canvas.get_buffer().raw, self.width(), self.height(), QImage.Format.Format_RGB32)

        fps = int(self.clock.get_fps())
        particle_count = self.simulation.n_particles
        self.setWindowTitle(f"Chladni Plate Simulation | FPS: {fps} | Particles: {particle_count}")

        self.update()

    def paintEvent(self, event: Any) -> None:
        QPainter(self).drawImage(0, 0, self.image)


class ControlWindow(QWidget):
    def __init__(self, simulation: Simulation) -> None:
        super().__init__()

        mu_label = QLabel("Mu:")

        mu_box = QDoubleSpinBox()
        mu_box.setKeyboardTracking(False)
        mu_box.setRange(0, 1)

        frequency_label = QLabel("Frequency:")

        frequency_box = QDoubleSpinBox()
        frequency_box.setKeyboardTracking(False)
        frequency_box.setRange(0, 0)

        frequency_slider = QSlider()
        frequency_slider.setOrientation(QtCore.Qt.Orientation.Horizontal)
        frequency_slider.setRange(0, 0)

        particles_label = QLabel("Particles:")

        particles_box = QSpinBox()
        particles_box.setKeyboardTracking(False)
        particles_box.setRange(0, 10000)

        throw_particles_button = QPushButton("Throw Particles")

        keep_throwing_particles_button = QPushButton("Keep Throwing Particles")
        keep_throwing_particles_button.setCheckable(True)

        def set_mu(mu: float) -> None:
            if mu == simulation.mu:
                return
            mu = simulation.set_mu(mu)
            mu_box.setValue(mu)
            frequency_box.setRange(simulation.min_frequency, simulation.max_frequency)
            frequency_slider.setRange(round(simulation.min_frequency * 100), round(simulation.max_frequency * 100))

        mu_box.valueChanged.connect(lambda value: set_mu(value))

        def set_frequency(frequency: float) -> None:
            if frequency == simulation.frequency:
                return
            frequency = simulation.set_frequency(frequency)
            frequency_box.setValue(frequency)

        frequency_box.valueChanged.connect(lambda value: set_frequency(value))

        frequency_box.valueChanged.connect(lambda value: frequency_slider.setValue(round(value * 100)))
        frequency_slider.valueChanged.connect(lambda value: frequency_box.setValue(value / 100))

        throw_particles_button.clicked.connect(lambda: simulation.throw_particles(particles_box.value()))

        throw_particles_timer = QTimer(self)
        throw_particles_timer.timeout.connect(lambda: simulation.throw_particles(particles_box.value()))
        keep_throwing_particles_button.toggled.connect(lambda toggled: throw_particles_timer.start(500) if toggled else throw_particles_timer.stop())

        layout = QVBoxLayout()
        layout.addWidget(mu_label)
        layout.addWidget(mu_box)
        layout.addWidget(frequency_label)
        layout.addWidget(frequency_box)
        layout.addWidget(frequency_slider)
        layout.addWidget(particles_label)
        layout.addWidget(particles_box)
        layout.addWidget(throw_particles_button)
        layout.addWidget(keep_throwing_particles_button)
        self.setLayout(layout)

        self.setWindowTitle(" ")

        mu_box.setValue(0.5)
        frequency_box.setValue(-np.inf)
        particles_box.setValue(100)


def simulate_plate() -> None:
    pygame.init()
    simulation = Simulation()
    application = QApplication(sys.argv)
    simulation_window = SimulationWindow(simulation)
    simulation_window.show()
    control_window = ControlWindow(simulation)
    control_window.show()
    application.exec()


def show_patterns(frequency_ratio: float, n_patterns: int, grid_size: int = 50, mu: float = 0.2, mode: Literal["full", "contour"] = "contour") -> None:
    (patterns, frequencies) = chladni_maths.calculate_patterns(grid_size, mu, frequency_ratio, n_patterns)
    plt.figure()
    for index in range(n_patterns):
        plt.subplot(round(sqrt(n_patterns)) + 1, round(sqrt(n_patterns)) + 1, index + 1)
        plt.title(f"Eigenfrequenz = {frequencies[index]:.2f}")
        match mode:
            case "full":
                plt.imshow(patterns[index])
            case "contour":
                plt.contour(patterns[index], (-1e-10, +1e-10))
            case _:
                assert False
    plt.show()
