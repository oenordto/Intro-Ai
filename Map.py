import numpy as np
import pandas as pd
from PIL import Image
from typing import Union


np.set_printoptions(threshold=np.inf, linewidth=300)

# Original code by Håkon Måløy
# Extended and documented by Xavier Sánchez-Díaz


class Map_Obj():
    """
    A map object helper class.

    Instantiate with a task number to create a single task. See the
    constructor information. Additionally, some getters are provided
    below. You can read more about them in their corresponding
    docstrings.

    Methods
    ----------
    get_cell_value(pos)
        Return the value (cost) of `pos`
    get_start_pos()
        Get the starting position of current task
    get_goal_pos()
        Get the goal position of current task
    get_end_goal_pos()
        Get the end goal position (for moving task)
    get_maps()
        Get integer and string maps
    """

    def __init__(self, task: int = 1) -> None:
        """Instantiate a map object for task number `task`.

        Parameters
        ----------
        task : int, optional
            Number of map / task to solve, by default task 1
        """
        self.start_pos, self.goal_pos, self.end_goal_pos, \
            self.path_to_map = self.fill_critical_positions(task)
        self.int_map, self.str_map = self.read_map(self.path_to_map)
        self.tmp_cell_value = self.get_cell_value(self.goal_pos)
        self.set_cell_value(self.start_pos, ' S ')
        self.set_cell_value(self.goal_pos, ' G ')
        self.tick_counter = 0

    def read_map(self, path: str) -> tuple[np.ndarray, str]:
        """
        Reads maps specified in path from file, converts them to numpy
        array and a string array. Then replaces specific values in the
        string array with predefined values more suitable for printing.

        Parameters
        ----------
        path : str
            Path to the map file (CSV)

        Returns
        -------
        tuple[np.ndarray, str]
            A tuple of the map as an ndarray of integers,
            and the map as a string of symbols.
        """
        # Read map from provided csv file
        df = pd.read_csv(path, index_col=None,
                         header=None)  # ,error_bad_lines=False)
        # Convert pandas dataframe to numpy array
        data = df.values
        # Convert numpy array to string to make it more human readable
        data_str = data.astype(str)
        # Replace numeric values with more human readable symbols
        data_str[data_str == '-1'] = ' # '
        data_str[data_str == '1'] = ' . '
        data_str[data_str == '2'] = ' , '
        data_str[data_str == '3'] = ' : '
        data_str[data_str == '4'] = ' ; '
        return data, data_str

    def fill_critical_positions(self, task: int) -> tuple[list[int], list[int],
                                                          list[int], str]:
        """
        Fill the important positions for the current task. Given the
        task, the path to the correct map is set, and the start, goal
        and eventual end_goal positions are set.

        Parameters
        ----------
        task : int
            Number of task we are currently solving

        Returns
        -------
        tuple[list[int], list[int], list[int], str]
            Start position
            Initial goal position
            End goal position
            Path to map for current task
        """
        if task == 1:
            start_pos = [27, 18]
            goal_pos = [40, 32]
            end_goal_pos = goal_pos
            path_to_map = 'Samfundet_map_1.csv'
        elif task == 2:
            start_pos = [40, 32]
            goal_pos = [8, 5]
            end_goal_pos = goal_pos
            path_to_map = 'Samfundet_map_1.csv'
        elif task == 3:
            start_pos = [28, 32]
            goal_pos = [6, 32]
            end_goal_pos = goal_pos
            path_to_map = 'Samfundet_map_2.csv'
        elif task == 4:
            start_pos = [28, 32]
            goal_pos = [6, 32]
            end_goal_pos = goal_pos
            path_to_map = 'Samfundet_map_Edgar_full.csv'
        elif task == 5:
            start_pos = [14, 18]
            goal_pos = [6, 36]
            end_goal_pos = [6, 7]
            path_to_map = 'Samfundet_map_2.csv'

        return start_pos, goal_pos, end_goal_pos, path_to_map

    def get_cell_value(self, pos: list[int, int]) -> int:
        """Getter for the value (cost) of the cell at `pos`"""
        return self.int_map[pos[0], pos[1]]

    def get_goal_pos(self) -> list[int, int]:
        """Getter for the goal position of the current task"""
        return self.goal_pos

    def get_start_pos(self):
        """Getter for the starting position of the current task"""
        return self.start_pos

    def get_end_goal_pos(self):
        """Getter for the end goal position of the moving task"""
        return self.end_goal_pos

    def get_maps(self) -> tuple[np.ndarray, str]:
        """Getter for the maps in both integer and string form"""
        # Return the map in both int and string format
        return self.int_map, self.str_map

    def move_goal_pos(self, pos: list[int, int]):
        """
        Moves the goal position towards `pos`. Moves the current goal
        position and replaces its previous position with the previous
        values for correct printing.

        Parameters
        ----------
        pos : list[int, int]
            New position of the goal
        """
        tmp_val = self.tmp_cell_value
        tmp_pos = self.goal_pos
        self.tmp_cell_value = self.get_cell_value(pos)
        self.goal_pos = [pos[0], pos[1]]
        self.replace_map_values(tmp_pos, tmp_val, self.goal_pos)

    def set_cell_value(self, pos: list[int, int], value: int,
                       str_map: bool = True):
        """Helper function to set the `value` of the cell at `pos`

        Parameters
        ----------
        pos : list[int, int]
            Position of cell to be updated
        value : int
            New value (cost) of the cell
        str_map : bool, optional
            A flag to know which map to update. By default, the
            string map is updated.
        """
        if str_map:
            self.str_map[pos[0], pos[1]] = value
        else:
            self.int_map[pos[0], pos[1]] = value

    def print_map(self, map_to_print: Union[np.ndarray, str]):
        """Helper function to print `map_to_print` in the console"""
        for column in map_to_print:
            print(column)

    def pick_move(self) -> list[int, int]:
        """
        Calculate new end_goal position based on the current position.

        Returns
        -------
        pos : list[int, int]
            New position of the goal.
        """
        if self.goal_pos[0] < self.end_goal_pos[0]:
            return [self.goal_pos[0] + 1, self.goal_pos[1]]
        elif self.goal_pos[0] > self.end_goal_pos[0]:
            return [self.goal_pos[0] - 1, self.goal_pos[1]]
        elif self.goal_pos[1] < self.end_goal_pos[1]:
            return [self.goal_pos[0], self.goal_pos[1] + 1]
        else:
            return [self.goal_pos[0], self.goal_pos[1] - 1]

    def replace_map_values(self, pos: list[int, int], value: int,
                           goal_pos: list[int, int]):
        """Replaces the values of the coordinates provided in
        both maps (int and str).

        Parameters
        ----------
        pos : list[int, int]
            Coordinates for where we want to change the values
        value : int
            The value we want to change to
        goal_pos : list[int, int]
            Coordinates of the current goal
        """
        if value == 1:
            str_value = ' . '
        elif value == 2:
            str_value = ' , '
        elif value == 3:
            str_value = ' : '
        elif value == 4:
            str_value = ' ; '
        else:
            str_value = str(value)
        self.int_map[pos[0]][pos[1]] = value
        self.str_map[pos[0]][pos[1]] = str_value
        self.str_map[goal_pos[0], goal_pos[1]] = ' G '

    def tick(self) -> list[int, int]:
        """
        Moves the current goal position every 4th call if current goal
        position is not already at the end_goal position.

        Returns
        -------
        pos : list[int, int]
            New position of the goal.
        """
        # For every 4th call, actually do something
        if self.tick_counter % 4 == 0:
            # The end_goal_pos is not set
            if self.end_goal_pos is None:
                return self.goal_pos
            # The current goal is at the end_goal
            elif self.end_goal_pos == self.goal_pos:
                return self.goal_pos
            else:
                # Move current goal position
                move = self.pick_move()
                self.move_goal_pos(move)
                # print(self.goal_pos)
        self.tick_counter += 1

        return self.goal_pos

    def set_start_pos_str_marker(self, start_pos: list[int, int],
                                 themap: Union[np.ndarray, str]):
        """Sets the start position marker at `start_pos` in `map`

        Parameters
        ----------
        start_pos : list[int, int]
            Position which we want to mark as the start
        themap : np.ndarray or str
            Map in which we want to change the starting position
        """
        # Attempt to set the start position on the map
        if self.int_map[start_pos[0]][start_pos[1]] == -1:
            self.print_map(self.str_map)
            print('The selected start position, ' + str(start_pos) +
                  ' is not a valid position on the current map.')
            exit()
        else:
            themap[start_pos[0]][start_pos[1]] = ' S '

    def set_goal_pos_str_marker(self, goal_pos: list[int, int],
                                themap: Union[np.ndarray, str]):
        """Set the goal position marker at `goal_pos` in `map`

        Parameters
        ----------
        goal_pos : list[int, int]
            Position which we want to mark as the goal
        themap : np.ndarray or str
            Map in which we want to change the goal position
        """
        # Attempt to set the goal position on the map
        if self.int_map[goal_pos[0]][goal_pos[1]] == -1:
            self.print_map(self.str_map)
            print('The selected goal position, ' + str(goal_pos) +
                  ' is not a valid position on the current map.')
            exit()
        else:
            themap[goal_pos[0]][goal_pos[1]] = ' G '

    def show_map(self, path=None, themap=None):
        """Draws `themap` as an image and shows it.

        Parameters
        ----------
        themap : np.ndarray or str, optional
            The map to show. By default uses the string map.
        path : list of tuple, optional
            List of coordinates representing the path to be displayed.
        """
        if themap is not None:
            self.set_start_pos_str_marker(self.start_pos, themap)
            self.set_goal_pos_str_marker(self.goal_pos, themap)
        else:
            themap = self.str_map

        if path is not None:
            for pos in path:
                themap[pos[0]][pos[1]] = ' Y '

        width = themap.shape[1]
        height = themap.shape[0]
        scale = 20
        image = Image.new(
            'RGB', (width * scale, height * scale), (255, 255, 0))
        pixels = image.load()

        colors = {
            ' # ': (211, 33, 45),
            ' . ': (215, 215, 215),
            ' , ': (166, 166, 166),
            ' : ': (96, 96, 96),
            ' ; ': (36, 36, 36),
            ' S ': (255, 0, 255),
            ' G ': (0, 128, 255),
            ' Y ': (255, 255, 0)  # Yellow for path
        }

        for y in range(height):
            for x in range(width):
                if themap[y][x] not in colors:
                    continue
                for i in range(scale):
                    for j in range(scale):
                        pixels[x * scale + i, y * scale +
                               j] = colors[themap[y][x]]

        image.show()

    def a_star(self, task):
        file = task[3]
        read_the_file = self.read_map(file)
        start_pos = ((task[0][0], task[0][1]),
                     read_the_file[0][task[0][0]][task[0][1]])
        end_goal_pos = tuple(task[2])

        open_set = [start_pos]  # Nodes to be evaluated
        closed_set = set()  # Nodes already evaluated

        g_score = {}  # Dictionary to store the g-scores
        g_score[start_pos[0]] = 0  # Cost from start to start is 0

        f_score = {}  # Dictionary to store the f-scores
        f_score[start_pos[0]] = self.heuristic(start_pos[0], end_goal_pos)
        parents = {}

        while open_set:
            # Find the node with the lowest f-score in the open set
            current = min(
                open_set, key=lambda x: f_score.get(x[0], float('inf')))

            # If the current node equals the end goal, end the loop and start recostructing the path
            if current[0] == end_goal_pos:
                return self.reconstruct_path(parents, current[0])

            # Delete from the list where nodes will be evaluated and add the coordinates to the list where nodes are evaluated
            open_set.remove(current)
            closed_set.add(current[0])

            # Find the neighbours of the current node
            neighbors = self.find_neighbour(
                current[0][0], current[0][1], read_the_file)

            # Iterate over the returned list of neighbours.
            for data in neighbors:
                neighbor = data[0]
                move_cost = data[1]
                parent = data[2]

                if neighbor in closed_set:
                    continue

                # Gets the current g_score
                tentative_g_score = g_score.get(
                    current[0], float('inf')) + move_cost

                # If current tentative g_score < g_score to the neighbour.
                #  Replace the g_score of the neighbour and calculate the f_score with the heuristic function
                if tentative_g_score < g_score.get(neighbor, float('inf')):
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + \
                        self.heuristic(neighbor, end_goal_pos)

                    if neighbor not in open_set:
                        # Appened the neighbour to the open_set if not in there
                        open_set.append((neighbor, move_cost))
                        # Adds the neigbhour and the parent to keep track of the path
                        parents[neighbor] = parent

        return None

    # The reconstruction of the path.
    # Returns a list of coordinates.

    def reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        # Pops the first and the last node since it is the start node and the end goal
        path.pop(-1)
        path.pop(0)
        return path

    # The heuristic funtion, Manhatten distance
    def heuristic(self, current_pos, end_goal_pos):
        return abs(current_pos[0] - end_goal_pos[0]) + abs(current_pos[1] - end_goal_pos[1])

    def find_neighbour(self, x_value, y_value, file):
        neigbhours = []

        # Calculate the move cost for each direction
        left_cost = file[0][x_value - 1][y_value]
        right_cost = file[0][x_value + 1][y_value]
        up_cost = file[0][x_value][y_value - 1]
        down_cost = file[0][x_value][y_value + 1]

        # Add neighbors to the list with their coordinates, move cost, and parent coordinates
        if left_cost > 0:
            neigbhours.append(
                ((x_value - 1, y_value), left_cost, (x_value, y_value)))

        if right_cost > 0:
            neigbhours.append(
                ((x_value + 1, y_value), right_cost, (x_value, y_value)))

        if up_cost > 0:
            neigbhours.append(
                ((x_value, y_value - 1), up_cost, (x_value, y_value)))

        if down_cost > 0:
            neigbhours.append(
                ((x_value, y_value + 1), down_cost, (x_value, y_value)))

        # Return a list of neighbours
        return neigbhours


map = Map_Obj(task=1)
map.show_map(path=map.a_star(map.fill_critical_positions(1)))
