# visualization.py
import matplotlib.pyplot as plt
import numpy as np

class Visualization:
    """Handles real-time plotting of the map, car, and path."""
    def __init__(self, width, length, sampling):
        self.fig, self.ax = plt.subplots(figsize=(7, 10))
        self.width = width
        self.length = length
        self.sampling = sampling
        
        # Set up plot
        self.ax.set_title("Live Map")
        self.ax.set_xlabel("Width (cm)")
        self.ax.set_ylabel("Length (cm)")
        self.ax.set_xlim(0, self.width)
        self.ax.set_ylim(0, self.length)
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        # Artists for updating plot
        self.map_img = self.ax.imshow(np.zeros((self.length, self.width)), cmap='gray_r', origin='lower', extent=[0, self.width, 0, self.length])
        self.car_artist, = self.ax.plot([], [], 'ro', markersize=10, label="Car")
        self.path_artist, = self.ax.plot([], [], 'g-', linewidth=2, label="Path")
        self.goal_artist, = self.ax.plot([], [], 'g*', markersize=15, label="Goal")
        
        self.ax.legend()
        plt.ion()
        plt.show()

    def update(self, map_grid, car_pos, path, goal_pos):
        """Update the plot with new data."""
        # Update map
        self.map_img.set_data(map_grid)
        
        # Update car position
        car_x, car_y, _ = car_pos
        self.car_artist.set_data([car_x * self.sampling], [car_y * self.sampling])
        
        # Update path
        if path:
            path_x = [p[0] * self.sampling for p in path]
            path_y = [p[1] * self.sampling for p in path]
            self.path_artist.set_data(path_x, path_y)
        else:
            self.path_artist.set_data([], [])
            
        # Update goal
        goal_x, goal_y, _ = goal_pos
        self.goal_artist.set_data([goal_x * self.sampling], [goal_y * self.sampling])
        
        # Redraw canvas
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def close(self):
        plt.ioff()
        plt.close(self.fig)
