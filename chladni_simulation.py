from functools import cached_property

import numpy as np
import pygame
from pygame import Surface
from scipy.interpolate import RegularGridInterpolator

import chladni_maths


def polar_to_cartesian(radius: np.ndarray, angle: np.ndarray) -> np.ndarray:
    return np.stack([radius * np.cos(angle), radius * np.sin(angle)], axis=-1)


class Simulation:
    # Hyperparameters
    grid_size: int = 50
    frequency_ratio: float = 8.0
    n_patterns: int = 30
    particle_size: int = 1
    max_age: float = np.inf
    age_milestones: np.ndarray = np.asarray([2000, 7000, 12000])
    color_milestones: np.ndarray = np.asarray([[0, 255, 0], [0, 0, 255], [255, 0, 0]])

    # Variables
    frequencies: np.ndarray | None
    patterns: np.ndarray | None
    mu: float | None
    frequency: float | None

    # Attributes
    pattern: np.ndarray
    particle_locations: np.ndarray
    particle_speeds: np.ndarray
    particle_directions: np.ndarray
    particle_ages: np.ndarray

    def __init__(self) -> None:
        self.frequencies = None
        self.patterns = None

        self.mu = None
        self.frequency = None
        self.pattern = np.ones(self.pattern_size)
        self.particle_locations = np.zeros((0, 2))
        self.particle_speeds = np.zeros(0)
        self.particle_directions = np.zeros(0)
        self.particle_ages = np.zeros(0)

    @cached_property
    def pattern_size(self) -> int:
        return self.grid_size**2

    @cached_property
    def grid_points(self) -> np.ndarray:
        return np.linspace(0, 1, self.grid_size)

    @property
    def n_particles(self) -> int:
        return len(self.particle_locations)

    @property
    def min_frequency(self) -> float:
        assert self.frequencies is not None
        return np.min(self.frequencies)

    @property
    def max_frequency(self) -> float:
        assert self.frequencies is not None
        return np.max(self.frequencies)

    def scale_pattern_to_speed(self, pattern: np.ndarray) -> np.ndarray:
        y = pattern
        y = np.abs(y)
        y = y / np.mean(y)
        y = (y + 0.1)**3
        y = y * 0.3
        y = y / 1000
        speeds = y
        return speeds

    def update_particle_speeds(self) -> None:
        self.particle_speeds = RegularGridInterpolator((self.grid_points, self.grid_points), np.reshape(self.scale_pattern_to_speed(self.pattern), shape=(self.grid_size, self.grid_size)))(self.particle_locations)
        self.particle_directions = np.random.uniform(low=0, high=2 * np.pi, size=self.n_particles)

    def set_mu(self, mu: float) -> float:
        mu = np.clip(mu, 0.01, 0.99)
        (self.patterns, self.frequencies) = chladni_maths.calculate_patterns(self.grid_size, mu, self.frequency_ratio, self.n_patterns)
        self.mu = mu

        if self.frequency is not None:
            self.set_frequency(self.frequency)

        return mu

    def set_frequency(self, frequency: float) -> float:
        if self.frequencies is None:
            return frequency

        assert self.patterns is not None
        assert self.frequencies is not None

        frequency = np.clip(frequency, self.min_frequency, self.max_frequency)

        right_index = np.argmax(self.frequencies >= frequency)
        left_index = right_index - 1

        left_frequency = self.frequencies[left_index]
        right_frequency = self.frequencies[right_index]
        left_pattern = self.patterns[left_index]
        right_pattern = self.patterns[right_index]

        t = 2 * (frequency - left_frequency) / (right_frequency - left_frequency) - 1
        assert -1 <= t and t <= +1

        if np.abs(t) < 0.5:
            self.pattern = np.ones(self.pattern_size)
        else:
            t = np.sign(t) * (np.abs(t) - 0.5) * 2
            t = (t + 1) / 2
            self.pattern = t * right_pattern + (1 - t) * left_pattern

        self.frequency = frequency

        return frequency

    def throw_particles(self, n_particles: int) -> None:
        new_particle_locations = np.random.uniform(low=0, high=1, size=(n_particles, 2))
        new_particle_speeds = np.zeros(n_particles)
        new_particle_directions = np.zeros(n_particles)
        new_particle_ages = np.zeros(n_particles)

        self.particle_locations = np.concatenate([self.particle_locations, new_particle_locations], axis=0)
        self.particle_speeds = np.concatenate([self.particle_speeds, new_particle_speeds])
        self.particle_directions = np.concatenate([self.particle_directions, new_particle_directions])
        self.particle_ages = np.concatenate([self.particle_ages, new_particle_ages])

    def step(self, dt: float) -> None:
        self.particle_locations += dt * polar_to_cartesian(radius=self.particle_speeds, angle=self.particle_directions)
        self.particle_ages += dt

        keep_bidx = np.all((0 <= self.particle_locations) & (self.particle_locations <= 1), axis=-1) & (self.particle_ages <= self.max_age)

        self.particle_locations = self.particle_locations[keep_bidx]
        self.particle_speeds = self.particle_speeds[keep_bidx]
        self.particle_directions = self.particle_directions[keep_bidx]
        self.particle_ages = self.particle_ages[keep_bidx]

    def draw(self, canvas: Surface) -> None:
        particle_locations_on_screen = self.particle_locations * np.asarray(canvas.get_size())
        particle_colors = RegularGridInterpolator((self.age_milestones,), self.color_milestones)(np.clip(self.particle_ages, self.age_milestones[0], self.age_milestones[-1]))
        canvas.fill("black")
        for (particle_location_on_screen, particle_color) in zip(particle_locations_on_screen, particle_colors):
            pygame.draw.circle(canvas, particle_color, particle_location_on_screen, self.particle_size)
