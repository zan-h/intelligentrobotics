from controller import Robot, Receiver, Emitter
import sys,struct,math
import numpy as np
import mlp as ntw

class Controller:
    def __init__(self, robot):        
        # robot parameters that should not be changed
        self.robot = robot
        self.time_step = 32 # ms
        self.max_speed = 1  # m/s
 
        # mlp configuration
        # input layer has 11 inputs: 3 ground sensors + 4 proximity sensors + 4 light sensors
        self.number_input_layer = 11
        self.number_hidden_layer = [12, 8]
        self.number_output_layer = 2
        
        # create list of neurons per layer
        self.number_neuros_per_layer = []
        self.number_neuros_per_layer.append(self.number_input_layer)
        self.number_neuros_per_layer.extend(self.number_hidden_layer)
        self.number_neuros_per_layer.append(self.number_output_layer)
        
        # initialize neural network
        self.network = ntw.MLP(self.number_neuros_per_layer)
        self.inputs = []
        
        # calculate total number of weights in mlp
        self.number_weights = 0
        for n in range(1,len(self.number_neuros_per_layer)):
            if(n == 1):
                # input + bias
                self.number_weights += (self.number_neuros_per_layer[n-1]+1)*self.number_neuros_per_layer[n]
            else:
                self.number_weights += self.number_neuros_per_layer[n-1]*self.number_neuros_per_layer[n]

        # enable and initialize motors
        self.left_motor = self.robot.getDevice('left wheel motor')
        self.right_motor = self.robot.getDevice('right wheel motor')
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)
        self.velocity_left = 0
        self.velocity_right = 0
    
        # enable proximity sensors
        self.proximity_sensors = []
        for i in range(8):
            sensor_name = 'ps' + str(i)
            self.proximity_sensors.append(self.robot.getDevice(sensor_name))
            self.proximity_sensors[i].enable(self.time_step)
       
        # enable ground sensors
        self.left_ir = self.robot.getDevice('gs0')
        self.left_ir.enable(self.time_step)
        self.center_ir = self.robot.getDevice('gs1')
        self.center_ir.enable(self.time_step)
        self.right_ir = self.robot.getDevice('gs2')
        self.right_ir.enable(self.time_step)
        
        # initialize light sensors using proximity sensors in light mode
        self.light_sensors = []
        for i in range(8):
            sensor_name = 'ls' + str(i)
            sensor = self.robot.getDevice(sensor_name)
            if sensor is None:
                sensor = self.proximity_sensors[i]
            sensor.enable(self.time_step)
            self.light_sensors.append(sensor)
        
        # disable ir leds to use proximity sensors in light mode
        self.leds = []
        for i in range(8):
            led_name = 'led' + str(i)
            led = self.robot.getDevice(led_name)
            if led is not None:
                led.set(0)
                self.leds.append(led)
        
        # enable communication with supervisor
        self.emitter = self.robot.getDevice("emitter") 
        self.receiver = self.robot.getDevice("receiver") 
        self.receiver.enable(self.time_step)
        self.receivedData = "" 
        self.receivedDataPrevious = "" 
        self.flagMessage = False
        
        # reward zone configuration
        self.reward_zone = {
            'x': 0.37,
            'y': -0.2,
            'z': -0.17,
            'threshold': 0.1
        }
        
        # fitness tracking
        self.fitness_values = []
        self.fitness = 0
        
        # light sensor history for detecting changes
        self.light_sensor_history = [[] for _ in range(8)]
        self.buffer_size = 10
    
    def check_for_new_genes(self):
        """
        receives and processes new genes from supervisor:
        - receives gene string and converts to float array
        - reshapes genes into weight matrices for each layer
        - updates neural network weights
        """
        while self.receiver.getQueueLength() > 0:
            genes = self.receiver.getString()
            
            # convert string "[1.0, 2.0, 3.0, ...]" to float array
            genes = genes[1:-1]
            genes = genes.split()
            genes = [float(gene.strip(',')) for gene in genes]
            
            # calculate sizes for weight matrices
            size1 = (self.number_neuros_per_layer[0] + 1) * self.number_neuros_per_layer[1]
            size2 = (self.number_neuros_per_layer[1] + 1) * self.number_neuros_per_layer[2]
            size3 = (self.number_neuros_per_layer[2] + 1) * self.number_neuros_per_layer[3]

            # split genes into weight matrices
            weights1 = np.array(genes[0:size1])
            weights2 = np.array(genes[size1:size1+size2])
            weights3 = np.array(genes[size1+size2:])

            # reshape and update network weights
            weights1 = weights1.reshape(self.number_neuros_per_layer[0] + 1, self.number_neuros_per_layer[1])
            weights2 = weights2.reshape(self.number_neuros_per_layer[1] + 1, self.number_neuros_per_layer[2])
            weights3 = weights3.reshape(self.number_neuros_per_layer[2], self.number_neuros_per_layer[3])

            self.network.weights[0] = weights1
            self.network.weights[1] = weights2
            self.network.weights[2] = weights3

            self.receiver.nextPacket()

    def clip_value(self,value,min_max):
        if (value > min_max):
            return min_max;
        elif (value < -min_max):
            return -min_max;
        return value;

    def sense_compute_and_actuate(self):
        """
        main control loop that:
        - gets neural network output
        - processes sensor readings
        - implements motion control with:
          - straight line motion when safe
          - speed limiting
          - turn rate limiting
        - sets final motor velocities
        """
        output = self.network.propagate_forward(self.inputs)
        
        light_sensors = self.inputs[-4:]
        left_light = (light_sensors[0] + light_sensors[1]) / 2
        right_light = (light_sensors[2] + light_sensors[3]) / 2
        
        proximity_sensors = self.inputs[3:7]
        front_proximity = max(proximity_sensors[0], proximity_sensors[1])
        
        left_speed = output[0]
        right_speed = output[1]
        
        # go straight if path is clear
        light_difference = abs(left_light - right_light)
        if front_proximity < 0.3 and light_difference < 0.2:
            avg_speed = (left_speed + right_speed) / 2
            left_speed = avg_speed
            right_speed = avg_speed
        
        # apply speed limits
        max_speed = 0.7
        min_speed = 0.1
        left_speed = np.clip(left_speed, min_speed, max_speed)
        right_speed = np.clip(right_speed, min_speed, max_speed)
        
        # limit turn rate
        max_diff = 0.3
        speed_diff = left_speed - right_speed
        if abs(speed_diff) > max_diff:
            avg_speed = (left_speed + right_speed) / 2
            left_speed = avg_speed + (max_diff/2 * np.sign(speed_diff))
            right_speed = avg_speed - (max_diff/2 * np.sign(speed_diff))
        
        self.velocity_left = left_speed
        self.velocity_right = right_speed
        
        self.left_motor.setVelocity(self.velocity_left * 3)
        self.right_motor.setVelocity(self.velocity_right * 3)

    def calculate_fitness(self):
        """
        calculates overall fitness using weighted components:
        - forward motion (35%): rewards moving forward
        - wall avoidance (25%): rewards staying away from walls
        - stability (25%): penalizes spinning
        - light detection (15%): rewards detecting light on right side
        """
        forward_speed = (self.velocity_left + self.velocity_right) / 2.0
        forward_fitness = max(0, forward_speed)
        
        proximity_value = max(self.inputs[3:7])
        wall_avoidance = 1.0 - proximity_value
        
        spin_difference = abs(self.velocity_left - self.velocity_right)
        stability_fitness = np.exp(-5 * spin_difference)
        
        right_light_sensors = self.inputs[-2:]
        light_fitness = max(right_light_sensors)
        
        self.fitness = (
            0.35 * forward_fitness +
            0.25 * wall_avoidance +
            0.25 * stability_fitness +
            0.15 * light_fitness
        )
        
        self.fitness_values.append(self.fitness)
        self.fitness = np.mean(self.fitness_values)

    def handle_emitter(self):
        """sends weight count and fitness data to supervisor"""
        data = str(self.number_weights)
        data = "weights: " + data
        string_message = str(data)
        string_message = string_message.encode("utf-8")
        self.emitter.send(string_message)

        data = str(self.fitness)
        data = "fitness: " + data
        string_message = str(data)
        string_message = string_message.encode("utf-8")
        self.emitter.send(string_message)
            
    def handle_receiver(self):
        """processes incoming messages from supervisor and checks if genotype is new"""
        if self.receiver.getQueueLength() > 0:
            while(self.receiver.getQueueLength() > 0):
                self.receivedData = self.receiver.getString()
                self.receivedData = self.receivedData[1:-1]
                self.receivedData = self.receivedData.split()
                x = np.array(self.receivedData)
                self.receivedData = x.astype(float)
                self.receiver.nextPacket()
                
            self.flagMessage = not np.array_equal(self.receivedDataPrevious,self.receivedData)
            self.receivedDataPrevious = self.receivedData 
        else:
            self.flagMessage = False

    def run_robot(self):
        """
        main robot control loop that:
        - reads and normalizes sensor inputs
        - handles communication with supervisor
        - processes genetic algorithm updates
        - runs motion control and fitness calculation
        """
        while self.robot.step(self.time_step) != -1:
            self.inputs = []
            
            # read and normalize ground sensors
            left = self.left_ir.getValue()
            center = self.center_ir.getValue()
            right = self.right_ir.getValue()
            
            min_gs = 0
            max_gs = 1023
            self.inputs.extend([
                (left-min_gs)/(max_gs-min_gs),
                (center-min_gs)/(max_gs-min_gs),
                (right-min_gs)/(max_gs-min_gs)
            ])
            
            # read front and right proximity sensors
            selected_proximity = [0, 1, 2, 7]
            for i in selected_proximity:
                temp = self.proximity_sensors[i].getValue()
                min_ds = 0
                max_ds = 4095
                temp = np.clip(temp, min_ds, max_ds)
                self.inputs.append((temp-min_ds)/(max_ds-min_ds))
            
            # read front and right light sensors
            selected_light = [0, 1, 2, 7]
            for i in selected_light:
                value = self.light_sensors[i].getValue()
                min_ls = 0
                max_ls = 4095
                value = np.clip(value, min_ls, max_ls)
                normalized_value = value / max_ls
                self.inputs.append(normalized_value)
                
                # update light history
                self.light_sensor_history[i].append(normalized_value)
                if len(self.light_sensor_history[i]) > self.buffer_size:
                    self.light_sensor_history[i].pop(0)
            
            # debug output
            if self.robot.step(self.time_step) % 100 == 0:
                print(f"Number of inputs: {len(self.inputs)}")
                print(f"Ground sensors: {self.inputs[0:3]}")
                print(f"Proximity sensors: {self.inputs[3:7]}")
                print(f"Light sensors: {self.inputs[7:11]}")
            
            self.handle_emitter()
            self.handle_receiver()
            
            self.check_for_new_genes()
            self.sense_compute_and_actuate()
            self.calculate_fitness()
            
    def is_at_decision_point(self):
        """
        determines if robot is at t-maze junction by checking:
        - front sensors for open space
        - side sensors for walls
        includes debug output for sensor readings
        """
        front_left = self.proximity_sensors[7].getValue()
        front_right = self.proximity_sensors[0].getValue()
        left_side = self.proximity_sensors[5].getValue()
        right_side = self.proximity_sensors[2].getValue()
        
        proximity_threshold = 0.02
        wall_threshold = 0.10

        front_clear = (front_left < proximity_threshold and 
                      front_right < proximity_threshold)
        side_walls = (left_side > wall_threshold and 
                     right_side > wall_threshold)

        if self.robot.step(self.time_step) % 100 == 0:
            print("\nDecision Point Detection Details:")
            print(f"Raw Sensor Values:")
            print(f"  Front Left: {front_left:.4f}")
            print(f"  Front Right: {front_right:.4f}")
            print(f"  Left Side: {left_side:.4f}")
            print(f"  Right Side: {right_side:.4f}")
            print(f"Thresholds:")
            print(f"  Proximity threshold: {proximity_threshold}")
            print(f"  Wall threshold: {wall_threshold}")
            print(f"Conditions:")
            print(f"  Front Clear: {front_clear} ({front_left:.4f}, {front_right:.4f})")
            print(f"  Side Walls: {side_walls} ({left_side:.4f}, {right_side:.4f})")
            if front_clear and side_walls:
                print("🚨 DECISION POINT DETECTED! 🚨")

        return front_clear and side_walls

    def detect_blinking_light(self):
        """detects blinking light by analyzing variance in sensor readings"""
        variances = []
        for i in range(8):
            if len(self.light_sensor_history[i]) >= self.buffer_size:
                variance = np.var(self.light_sensor_history[i])
                variances.append(variance)

        if not variances:
            return False

        avg_variance = np.mean(variances)
        blinking_threshold = 0.001
        return avg_variance > blinking_threshold

    def is_turning_right(self):
        """checks if robot is making right turn based on wheel velocities"""
        return self.velocity_left > self.velocity_right

if __name__ == "__main__":
    my_robot = Robot()
    controller = Controller(my_robot)
    controller.run_robot()
