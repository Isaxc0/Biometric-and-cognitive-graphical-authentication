import statistics
from collections import defaultdict

import math
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


class Participant:
    def __init__(self, parti_id):
        self.STAGES = [1, 2, 3, 4]  # stage numbers
        self.id = parti_id  # participant ID
        self.features = dict(id=parti_id)  # dictionary of features
        self.xs = None  # all x coordinates
        self.ys = None  # all y coordinates
        self.times = None  # all time stamps
        self.flags = None  # all indications of connecting a node
        self.nodes = None  # all node positions (x, y)
        self.all_timings = []  # all times to complete each pattern

    def import_file_data(self):
        """Imports data from the data into the attributes
           """
        raw_data = {}
        for stage in self.STAGES:
            fh = open(self.id + "\\" + self.id + "_" + str(stage) + "_data.txt")
            rows = []
            for line in fh:
                sep = line.replace("\n", '').split(" ")
                rows.append(sep)
            fh.close()
            raw_data[stage] = rows
        self.xs = self.set_xs(raw_data)
        self.ys = self.set_ys(raw_data)
        self.times = self.set_times(raw_data)
        self.flags = self.set_flags(raw_data)
        self.nodes = self.set_nodes(raw_data)

    def set_xs(self, data):
        """Imports x coordinate data from the data files
        :arg
            data (list): all data from a file
        :return
            dictionary with stage number as key and list of x coordinates as data
        """
        dict_xs = {}
        for stage in self.STAGES:
            xs = []
            for row in data[stage]:
                if row[0] == "end":
                    xs.append(row[0])
                else:
                    xs.append(int(row[0]))

            dict_xs[stage] = xs
        return dict_xs

    def set_ys(self, data):
        """Imports y coordinate data from the data
        :arg
            data (list): all data from a file
        :return
            dictionary with stage number as key and list of y coordinates as data
        """
        dict_ys = {}
        for stage in self.STAGES:
            ys = []
            for row in data[stage]:
                if row[0] == "end":
                    ys.append(row[1])
                else:
                    ys.append(int(row[1]))
            dict_ys[stage] = ys
        return dict_ys

    def set_times(self, data):
        """Imports time data from the data
        :arg
            data (list): all data from a file
        :return
            dictionary with stage number as key and list of times as data
        """
        dict_times = {}
        for stage in self.STAGES:
            times = []
            for row in data[stage]:
                if row[0] == "end":
                    times.append(row[2])
                else:
                    times.append(int(row[2]))
            dict_times[stage] = times
        return dict_times

    def set_flags(self, data):
        """Imports node connection indication (flags) coordinate data from the data
        :arg
            data (list): all data from a file
        :return
            dictionary with stage number as key and list of flags coordinates as data
        """
        dict_flags = {}
        for stage in self.STAGES:
            flags = []
            for row in data[stage]:
                flags.append(str(row[3]))
            dict_flags[stage] = flags
        return dict_flags

    def set_nodes(self, data):
        """Imports x and y coordinate data of the nodes from the data
        :arg
            data (list): all data from a file
        :return
            dictionary with stage number as key and list of pairs of x and y coordinates as data
        """
        """ sets list of all node positions
           """
        dict_nodes = {}
        for stage in self.STAGES:
            nodes = []
            for row in data[stage]:
                if row[3] == "1":
                    nodes.append((int(row[0]), int(row[1])))
                if row[3] == "end":
                    break
            dict_nodes[stage] = nodes
        return dict_nodes

    def plot_data(self, stage):
        """Plots the coordinate data (drawing path) for all drawings for a stage
        :arg
            stage (int): the stage number
        """
        x = []
        y = []
        nodes_x = []
        nodes_y = []
        if stage == 1 or stage == 4:
            connection_number = 6
        if stage == 2 or stage == 3:
            connection_number = 7

        colours = ["green", "purple", "blue", "pink", "black", "orange", "gray"]  # colours for each connection
        labels = ["Connection 1", "Connection 2", "Connection 3", "Connection 4", "Connection 5", "Connection 6",
                  "Connection 7"]
        labels = labels[:connection_number]
        labels.append("Over shoot")
        colours = colours[:connection_number]
        colours.append("#a31525")

        counter = 0
        end_found = False
        for i in range(len(self.xs[stage])):
            if self.flags[stage][i] == "1" and not end_found:
                nodes_x.append(self.xs[stage][i])
                nodes_y.append(self.ys[stage][i])
            else:
                if self.xs[stage][i] == "end":  # colour is set to red past the last node (overshoot)
                    plt.scatter(x, y, c="#a31525", s=10, alpha=0.2)
                    x = []
                    y = []
                    counter = 0
                    end_found = True
                else:
                    x.append(self.xs[stage][i])
                    y.append(self.ys[stage][i])
                    if self.flags[stage][i + 1] == "1":  # if connected node found, plot data
                        plt.scatter(x, y, c=colours[counter], s=10, alpha=0.2)
                        counter += 1
                        x = []
                        y = []

        plt.plot(nodes_x, nodes_y, linewidth=3)

        handles = []
        for i in range(len(labels)):
            handles.append(mpatches.Patch(color=colours[i], label=labels[i]))

        plt.scatter(nodes_x[0], nodes_y[0], s=900, marker="o", color="#4ee20b")
        plt.scatter(nodes_x[len(nodes_x) - 1], nodes_y[len(nodes_y) - 1], s=900, marker="o", color="#ff1100")
        plt.scatter(nodes_x[1:len(nodes_x) - 1], nodes_y[1:len(nodes_y) - 1], s=900, marker="o")

        plt.title("Participant " + self.id + "'s drawing paths for stage " + str(stage))
        plt.ylabel("Y coordinate")
        plt.xlabel("X coordinate")
        plt.xticks(range(0, 800, 100))
        plt.yticks(range(100, 900, 100))
        plt.legend(labels, handles=handles, bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
        plt.gca().invert_yaxis()
        plt.xticks(rotation=0)
        plt.savefig(self.id + "_" + str(stage) + "_drawing_data.png", bbox_inches="tight")
        plt.clf()

    def separate_attempts(self, stage):
        """Separates each drawings data into individual elements in a list
        :arg
            stage (int): the stage number
        """
        drawings = []  # 2d list of all coordinates for each drawing
        attempt_data = []  # current coordinates from drawing
        for i in range(len(self.xs[stage])):
            if self.xs[stage][i] == "end":
                drawings.append(attempt_data)
                attempt_data = []
            else:
                attempt_data.append((self.xs[stage][i], self.ys[stage][i], self.times[stage][i], self.flags[stage][i]))

        return drawings

    def separate_connections(self, drawing):
        """Separates all coordinates between each connection made to a node
        :arg
            drawing (list): drawing data for that stage
        """
        connections = []  # 2d list with separated coordinates by each connection
        connection_data = []  # holds current coords for this connection
        for i in range(1, len(drawing)):
            if drawing[i][3] == "1":
                connections.append(connection_data)
                connection_data = []
            if drawing[i][0] == "end":  # stop when last node is connected
                break
            else:
                connection_data.append((drawing[i][0], drawing[i][1], drawing[i][2], drawing[i][3]))

        return connections

    def feature_displacements(self, stage):
        """Calculates the mean and maximum perpendicular distances from the connection lines
        :arg
            stage (int): the stage number
        """
        drawings = self.separate_attempts(stage)
        distances_means = defaultdict(list)  # key is the counter and the data is each distance from the two nodes
        distances_maxs = defaultdict(list)  # key is the counter and the data is each distance from the two nodes
        distances_mins = defaultdict(list)  # key is the counter and the data is each distance from the two nodes

        distances = []
        for drawing in drawings:
            connections = self.separate_connections(drawing)
            counter = 0
            for connection in connections:
                for data in connection:
                    p1 = (self.nodes[stage][counter][0], self.nodes[stage][counter][1])
                    p2 = (self.nodes[stage][counter + 1][0], self.nodes[stage][counter + 1][1])
                    p1 = np.asarray(p1)
                    p2 = np.asarray(p2)
                    p3 = np.asarray((data[0], data[1]))
                    distance = np.cross(p2 - p1, p3 - p1) / np.linalg.norm(p2 - p1)  # distance from the line
                    distances.append(distance)

                distances_means[counter].append(statistics.mean(distances))
                distances_maxs[counter].append(max(distances))
                distances_mins[counter].append(min(distances))
                distances = []
                counter += 1

        for i in range(len(distances_means)):
            self.features["mean_disp_" + str(stage) + "_" + str(i + 1)] = distances_means[i]
            self.features["max_disp_" + str(stage) + "_" + str(i + 1)] = distances_maxs[i]
            self.features["min_disp_" + str(stage) + "_" + str(i + 1)] = distances_mins[i]

    def feature_over_shoot(self, stage):
        """Calculates the mean distance of maximum distances that the participant overshoots past last node
        :arg
            stage (int): the stage number
        """
        last_node_coords = self.nodes[stage][len(self.nodes[stage]) - 1]  # position of the last node
        current_over_shoot_distances = []
        over_shoot_distances_maximum = []
        end = False
        for i in range(len(self.xs[stage])):
            if self.xs[stage][i] == last_node_coords[0] and self.ys[stage][i] == last_node_coords[1] and \
                    self.flags[stage][i] == "1":
                end = True

            if end and self.xs[stage][i] != "end":
                # calculate the distance from the last node
                distance = math.sqrt(
                    (self.xs[stage][i] - last_node_coords[0]) ** 2 + (self.ys[stage][i] - last_node_coords[1]) ** 2)
                current_over_shoot_distances.append(distance)

            if self.xs[stage][i] == "end":
                if current_over_shoot_distances:
                    over_shoot_distances_maximum.append(max(current_over_shoot_distances))

                current_over_shoot_distances = []
                end = False

        self.features["over_shoot_" + str(stage)] = over_shoot_distances_maximum  # save feature to all features

    def feature_timing(self, stage):
        """Calculates the mean timings between each set of nodes
        :arg
            stage (int): the stage number
        """
        start = True
        dict_times = defaultdict(list)  # key is the counter and the data is each distance from the two nodes
        key = 0
        for i in range(len(self.xs[stage])):
            if start:
                prev_time = self.times[stage][i]
                start = False
            else:
                #  calculate time take to draw whole connection
                if self.flags[stage][i] == "1":
                    time_difference = self.times[stage][i] - prev_time
                    prev_time = self.times[stage][i]
                    dict_times[key].append(time_difference)
                    key += 1

                if self.flags[stage][i] == "end":
                    key = 0
                    start = True

        for key, times in dict_times.items():
            self.features["time_" + str(stage) + "_" + str(key + 1)] = times  # save features to all features

    def feature_acceleration(self, stage):
        """Calculates the acceleration between each node
        :arg
            stage (int): the stage number
        """
        dict_acceration = defaultdict(list)  # key is the counter and the data is each distance from the two nodes
        initial_velocity = 0
        key = 0
        start = True
        for i in range(len(self.xs[stage])):
            if start:
                prev_node = [self.xs[stage][i], self.ys[stage][i]]
                prev_time = self.times[stage][i]
                start = False
            else:
                if self.flags[stage][i] == "1":
                    # calculate time, distance and velocity to calculate the acceleration
                    time = self.times[stage][i] - prev_time
                    distance = math.sqrt(
                        (self.xs[stage][i] - prev_node[0]) ** 2 + (self.ys[stage][i] - prev_node[1]) ** 2)
                    final_velocity = distance / time
                    acceleration = (final_velocity - initial_velocity) / time
                    prev_time = self.times[stage][i]
                    prev_node = [self.xs[stage][i], self.ys[stage][i]]
                    dict_acceration[key].append(acceleration)
                    key += 1

                if self.flags[stage][i] == "end":
                    key = 0
                    start = True

        for key, accelerations in dict_acceration.items():
            self.features["acc_" + str(stage) + "_" + str(key + 1)] = accelerations  # save features to all features

    def feature_curvature(self, stage):
        """Calculates the curvature between each node
        :arg
            stage (int): the stage number
        """
        curvature_means = defaultdict(list)
        curvature_maxs = defaultdict(list)
        drawings = self.separate_attempts(stage)
        for drawing in drawings:
            coordinates = []
            start = True
            key = 0
            for line in drawing:
                if start:
                    start = False
                else:
                    coordinates.append([line[0], line[1]])
                    if line[3] == "1":
                        coordinates_np = np.array(coordinates)
                        coordinates = [[line[0], line[1]]]

                        x = coordinates_np[:, 0]
                        y = coordinates_np[:, 1]

                        # first derivative
                        dx = np.gradient(x)
                        dy = np.gradient(y)

                        # second derivative
                        d2x = np.gradient(dx)
                        d2y = np.gradient(dy)

                        # calculation of curvature
                        curvature = np.abs(dx * d2y - d2x * dy) / (dx * dx + dy * dy) ** 1.5
                        curvature = curvature[~np.isnan(curvature)]

                        curvature_means[key].append(np.mean(curvature))
                        curvature_maxs[key].append(max(curvature))
                        key += 1
                        start = True

        # save features to all features
        for i in range(len(curvature_means)):
            self.features["max_curv_" + str(stage) + "_" + str(key + 1)] = curvature_means[i]
            self.features["mean_curv_" + str(stage) + "_" + str(key + 1)] = curvature_maxs[i]

    def generate_all_features(self, stage):
        """Generates all the features by calling correct methods
        :arg
            stage (int): the stage number
        """
        self.features = {"id": self.id}
        self.feature_displacements(stage)
        self.feature_over_shoot(stage)
        self.feature_timing(stage)
        self.feature_acceleration(stage)
        self.feature_curvature(stage)

        # get all time taken to draw each attempt for the participant for the stage
        stage_timings = []
        drawings = self.separate_attempts(stage)
        for drawing in drawings:
            stage_timings.append(drawing.pop()[2] - drawing[0][2])
        self.all_timings = stage_timings

    def remove_data(self, stage, remove_count):
        """Removes the specified number of drawings from the data

        :arg
            stage (int): the stage number
            remove_count (int): number of drawings to be removed
        """
        end_count = 0
        for i in range(len(self.xs[stage])):
            if self.xs[stage][i] == "end":
                end_count += 1
            if end_count >= remove_count:
                break

        self.xs[stage] = self.xs[stage][i + 1:]
        self.ys[stage] = self.ys[stage][i + 1:]
        self.times[stage] = self.times[stage][i + 1:]
        self.flags[stage] = self.flags[stage][i + 1:]

    def save_model(self, svm, dt, imposter):
        """Stores the tuned model to each participant with the key as the impsoter id

        :arg
            svm (SVM): tuned SVM model
            dt (DecisionTree): tuned Decision Tree model
            imposter (str) : ID of imposter participant that the models are tuned with
        """
        self.svms[imposter] = svm
        self.dts[imposter] = dt

    def chunking(self, stage):
        """Generates and prints the chunking from the participant
        :arg
            stage (int): the stage number
        """
        nodes = []  # list of tuple with structure (x, y)
        # get the centre of each node for stage
        for i in range(len(self.xs[stage])):
            if self.flags[stage][i] == "1":
                nodes.append((self.xs[stage][i], self.ys[stage][i]))
            else:
                if self.xs[stage][i] == "end":
                    break

        wait_times = defaultdict(
            list)  # dictionary with the node number as the key and time stayed within the node as the value
        for drawing in self.separate_attempts(stage):
            for i in range(len(nodes)):
                times_in_node = []
                initial_entry = False

                for j in range(len(drawing)):
                    x = drawing[j][0]
                    y = drawing[j][1]
                    in_node = self.in_node(x, y, nodes[i])

                    # if mouse is in node add the timestamp to all times
                    if in_node:
                        times_in_node.append(drawing[j][2])
                        if not initial_entry:
                            initial_entry = True

                    # calculate the time time in node by taking away the first time entered from the last time entered
                    if not in_node and initial_entry or i == len(nodes) - 1 and initial_entry:
                        time_in_node = max(times_in_node) - min(times_in_node)
                        wait_times[i].append(time_in_node)
                        break
        counter = 1
        for key, times in wait_times.items():
            counter += 1
            wait_times[key] = statistics.mean(times)  # get mean time across all drawings (50)

        wait_times = list(wait_times.values())

        chunks = []
        chunk = "1"
        prev_time = wait_times[0]
        # with the mean time in nodes, calculate the chunking
        for i in range(1, len(wait_times)):
            threshold = prev_time * 1.5
            chunk += str(i + 1)
            if wait_times[i] > threshold:  # if above the threshold new chunk begins
                chunks.append(chunk)
                chunk = ""
            if i + 1 == len(wait_times) and chunk != "":
                chunks.append(chunk)

            prev_time = wait_times[i]

        print(self.id, chunks)

    def in_node(self, x, y, node):
        """Checks whether mouse position is within the node

        :arg
            x (int): x coordinate of mouse position
            y (int): y coordinate of mouse position
            node ((int, int)) : tuple of x and y coordinate of the centre of the node
        :return
            bool: Returns true if mouse position is in the node else return false
        """
        node_x = node[0]
        node_y = node[1]
        width = 30
        if (node_x - width) < x < (node_x + width):
            if (node_y - width) < y < (node_y + width):
                return True  # position in node
        return False  # position not in node
