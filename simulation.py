from functools import cached_property
import random
import thorpy
import pygame
from thorpy.elements import Button, SliderWithText, Labelled, TextInput, ToggleButton, Box, Group, Text

import sys
from typing import Any, Literal
import numpy as np
import scipy.sparse as sp
from numpy import ndarray
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import scipy.sparse.linalg as sp_linalg
from scipy.interpolate import RegularGridInterpolator
import chladni
from pygame import Surface, Vector2

update_particle_speeds_event = pygame.USEREVENT + 1
set_mu_event = pygame.USEREVENT + 2
set_frequency_event = pygame.USEREVENT + 3
throw_particles_event = pygame.USEREVENT + 4


def polar_to_cartesian(radius: ndarray, angle: ndarray) -> ndarray:
    return np.stack([radius * np.cos(angle), radius * np.sin(angle)], axis=-1)


class Simulation:
    # Hyperparameters
    grid_size: int = 30
    n_patterns: int = 15
    update_particle_speeds_interval: int = 200
    particle_size: int = 1
    max_age: float = np.inf
    age_milestones: ndarray = np.asarray([2000, 7000, 12000])
    color_milestones: ndarray = np.asarray([[0, 255, 0], [0, 0, 255], [255, 0, 0]])

    # Variables
    frequencies: ndarray | None
    patterns: ndarray | None

    current_frequency: int
    current_pattern: ndarray

    particle_locations: ndarray
    particle_speeds: ndarray
    particle_directions: ndarray
    particle_ages: ndarray

    @cached_property
    def pattern_size(self) -> int:
        return self.grid_size**2

    @cached_property
    def grid_step(self) -> float:
        return 1 / (self.grid_size - 1)

    @cached_property
    def grid_points_1D(self) -> ndarray:
        return np.linspace(0, 1, self.grid_size)

    @cached_property
    def grid_points_2D(self) -> ndarray:
        return np.reshape(np.stack(np.broadcast_arrays(self.grid_points_1D[np.newaxis, :], self.grid_points_1D[:, np.newaxis]), axis=-1), shape=(self.grid_size**2, 2))

    def __init__(self) -> None:
        self.frequencies = None
        self.patterns = None

        self.current_pattern = np.ones(self.pattern_size)

        self.particle_locations = np.zeros((0, 2))
        self.particle_speeds = np.zeros(0)
        self.particle_directions = np.zeros(0)
        self.particle_ages = np.zeros(0)

    def set_mu(self, mu: float) -> None:  # exposed
        pygame.event.post(pygame.event.Event(set_mu_event, mu=mu))

    def set_frequency(self, frequency: int) -> None:  # exposed
        pygame.event.post(pygame.event.Event(set_frequency_event, frequency=frequency))

    def throw_particles(self, n_particles: int) -> None:  # exposed
        pygame.event.post(pygame.event.Event(throw_particles_event, n_particles=n_particles))

    def _scale_to_probabilities(self, pattern: ndarray) -> ndarray:  # unused
        y = pattern

        y = 1 / (y + 0.001)

        probabilities = y / np.sum(y)
        return probabilities

    def _generate_points(self, pattern: ndarray, n_points: int) -> ndarray:  # unused
        fidx = np.random.choice(a=np.arange(self.pattern_size), size=n_points, p=self._scale_to_probabilities(pattern))
        points = self.grid_points_2D[fidx, :] + polar_to_cartesian(
            radius=np.random.uniform(low=0, high=self.grid_step, size=n_points),
            angle=np.random.uniform(low=0, high=2 * np.pi, size=n_points)
        )
        return points

    def _points_to_fidx(self, points: ndarray) -> ndarray:  # unused
        idx = np.round(points * (self.grid_size - 1))
        fidx = idx[:, 0] + self.grid_size * idx[:, 1]
        return fidx

    def _scale_to_speeds(self, pattern: ndarray) -> ndarray:  # general
        y = pattern

        y = (y + 0.1)**3
        y = y / np.mean(y)
        y = y * 0.3

        speeds = y / 1000
        return speeds

    def _points_to_speeds(self, pattern: ndarray, points: ndarray) -> ndarray:  # general
        return RegularGridInterpolator((self.grid_points_1D, self.grid_points_1D), np.reshape(self._scale_to_speeds(pattern), shape=(self.grid_size, self.grid_size)))(points)

    def _get_n_particles(self) -> int:
        return self.particle_ages.size

    def _update_particle_speeds(self) -> None:
        self.particle_speeds = self._points_to_speeds(self.current_pattern, self.particle_locations)
        self.particle_directions = np.random.uniform(low=0, high=2 * np.pi, size=self._get_n_particles())

    def _set_mu(self, mu: float) -> None:
        assert 0 < mu and mu < 1

        (eigenvalues, eigenfunctions) = chladni.calculate_patterns(self.grid_size, mu, self.n_patterns)

        frequencies = np.sqrt(eigenvalues)
        patterns = np.abs(eigenfunctions)

        patterns = patterns / np.max(patterns)
        frequencies = (np.random.default_rng(seed=hash(mu)).integers(low=1, high=9)
                       + np.round((frequencies - frequencies[0]) / (frequencies[-1] - frequencies[0]) * 90))

        while np.any(np.diff(frequencies) == 0):
            overlap_idx = np.argwhere(np.diff(frequencies) == 0)
            frequencies[overlap_idx + 1] += 1

        self.frequencies = frequencies
        self.patterns = patterns

    def _set_frequency(self, frequency: int) -> None:
        assert 0 <= frequency and frequency <= 100
        assert self.frequencies is not None
        assert self.patterns is not None

        nearest_idx = np.argmin(np.abs(self.frequencies - frequency))
        nearest_frequency = self.frequencies[nearest_idx]
        nearest_pattern = self.patterns[nearest_idx]

        frequency_distance = np.abs(nearest_frequency - frequency)
        match frequency_distance:
            case 0:
                intensity = 1
            case 1:
                intensity = 0.9
            case 2:
                intensity = 0.8
            case 3:
                intensity = 0.7
            case _:
                intensity = 0

        self.current_frequency = frequency
        self.current_pattern = intensity * nearest_pattern + (1 - intensity) * np.ones(self.pattern_size)

    def _throw_particles(self, n_particles: int) -> None:
        # new_particle_locations = 0.5 + polar_to_cartesian(
        #     radius=np.random.uniform(low=0, high=0.3, size=n_particles),
        #     angle=np.random.uniform(low=0, high=2 * np.pi, size=n_particles)
        # )
        new_particle_locations = np.random.uniform(low=0, high=1, size=(n_particles, 2))
        new_particle_speeds = np.zeros(n_particles)
        new_particle_directions = np.zeros(n_particles)
        new_particle_ages = np.zeros(n_particles)

        self.particle_locations = np.concatenate([self.particle_locations, new_particle_locations], axis=0)
        self.particle_speeds = np.concatenate([self.particle_speeds, new_particle_speeds])
        self.particle_directions = np.concatenate([self.particle_directions, new_particle_directions])
        self.particle_ages = np.concatenate([self.particle_ages, new_particle_ages])

    def _simulation_step(self, dt: float) -> None:
        self.particle_locations += dt * polar_to_cartesian(radius=self.particle_speeds, angle=self.particle_directions)
        self.particle_ages += dt

        keep_bidx = np.all((0 < self.particle_locations) & (self.particle_locations < 1), axis=-1) & (self.particle_ages < self.max_age)

        self.particle_locations = self.particle_locations[keep_bidx]
        self.particle_speeds = self.particle_speeds[keep_bidx]
        self.particle_directions = self.particle_directions[keep_bidx]
        self.particle_ages = self.particle_ages[keep_bidx]

    def _draw_simulation(self, canvas: Surface) -> None:
        particle_locations_on_screen = self.particle_locations * np.asarray(canvas.get_size())
        particle_colors = RegularGridInterpolator((self.age_milestones,), self.color_milestones)(np.clip(self.particle_ages, self.age_milestones[0], self.age_milestones[-1]))

        canvas.fill("black")
        for (particle_location_on_screen, particle_color) in zip(particle_locations_on_screen, particle_colors):
            pygame.draw.circle(canvas, particle_color, particle_location_on_screen, self.particle_size)

    def start(self) -> None:  # exposed
        pygame.init()
        clock = pygame.time.Clock()
        screen = pygame.display.set_mode(size=(500, 500), flags=pygame.RESIZABLE)

        pygame.time.set_timer(update_particle_speeds_event, self.update_particle_speeds_interval)

        running = True
        while running:
            dt = clock.tick(60)
            fps = clock.get_fps()

            self.particle_locations += dt * polar_to_cartesian(radius=self.particle_speeds, angle=self.particle_directions)
            self.particle_ages += dt

            keep_bidx = np.all((0 < self.particle_locations) & (self.particle_locations < 1), axis=-1) & (self.particle_ages < self.max_age)

            self.particle_locations = self.particle_locations[keep_bidx]
            self.particle_speeds = self.particle_speeds[keep_bidx]
            self.particle_directions = self.particle_directions[keep_bidx]
            self.particle_ages = self.particle_ages[keep_bidx]

            pygame.display.set_caption(f"Particles: {self._get_n_particles()} | FPS: {fps}")

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == update_particle_speeds_event:
                    self._update_particle_speeds()
                if event.type == set_mu_event:
                    self._set_mu(event.mu)
                if event.type == set_frequency_event:
                    self._set_frequency(event.frequency)
                if event.type == throw_particles_event:
                    self._throw_particles(event.n_particles)

            particle_locations_on_screen = self.particle_locations * np.asarray(screen.get_size())
            particle_colors = RegularGridInterpolator((self.age_milestones,), self.color_milestones)(np.clip(self.particle_ages, self.age_milestones[0], self.age_milestones[-1]))

            screen.fill("black")
            for (particle_location_on_screen, particle_color) in zip(particle_locations_on_screen, particle_colors):
                pygame.draw.circle(screen, particle_color, particle_location_on_screen, self.particle_size)
            pygame.display.flip()

    def start_with_ui(self) -> None:  # exposed
        self._set_mu(0.5)
        self._set_frequency(0)

        pygame.init()
        clock = pygame.time.Clock()
        screen = pygame.display.set_mode(size=(800, 800), flags=pygame.RESIZABLE)

        thorpy.init(screen, thorpy.themes.theme_game2)

        mu_selector = TextInput("0.5")
        mu_selector.set_only_numbers()

        frequency_display = Text("")

        frequency_selector = TextInput("0")
        frequency_selector.set_only_integers()

        n_particles_selector = TextInput("100")
        n_particles_selector.set_only_integers()

        throw_button = Button("Throw Particles")

        keep_throwing_button = ToggleButton("Keep Throwing Particles")

        ui_elements = Group(
            [Group([Text("mu (between 0 and 1, exclusive):"), mu_selector, Text("Particles:"), n_particles_selector, throw_button, keep_throwing_button], mode="h"),
             Group([Text("Frequency (changable with LEFT/RIGHT key): "), frequency_display, Text("Set frequency (between 0 and 100, inclusive):"), frequency_selector], mode="h")],
            mode="v")

        mu_selector.on_validation = lambda: self.set_mu(float(mu_selector.get_value()))
        frequency_selector.on_validation = lambda: self.set_frequency(int(frequency_selector.get_value()))
        throw_button.at_unclick = lambda **params: self.throw_particles(int(n_particles_selector.get_value()))

        pygame.time.set_timer(update_particle_speeds_event, self.update_particle_speeds_interval)

        running = True
        while running:
            dt = clock.tick(60)
            fps = clock.get_fps()

            self._simulation_step(dt)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_LEFT:
                    self._set_frequency(max(self.current_frequency - 1, 0))
                if event.type == pygame.KEYDOWN and event.key == pygame.K_RIGHT:
                    self._set_frequency(min(self.current_frequency + 1, 100))
                if event.type == update_particle_speeds_event:
                    self._update_particle_speeds()
                if event.type == set_mu_event:
                    self._set_mu(event.mu)
                if event.type == set_frequency_event:
                    self._set_frequency(event.frequency)
                if event.type == throw_particles_event:
                    self._throw_particles(event.n_particles)

            frequency_display.set_text(str(self.current_frequency))

            simulation_canvas = Surface((screen.get_width(), screen.get_height() - 100))
            ui_canvas = Surface((screen.get_width(), 100))

            self._draw_simulation(simulation_canvas)
            ui_canvas.fill((100, 100, 100))

            screen.blit(simulation_canvas, (0, 0))
            screen.blit(ui_canvas, (0, screen.get_height() - 100))

            ui_elements.set_center(screen.get_width() / 2, screen.get_height() - 50)
            ui_elements.get_updater().update()

            pygame.display.set_caption(f"Particles: {self._get_n_particles()} | FPS: {fps}")
            pygame.display.flip()
