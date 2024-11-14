from controller import Supervisor
from controller import Keyboard
from controller import Display

import numpy as np
import ga,os,sys,struct

class SupervisorGA:
    def __init__(self):
        # Simulation Parameters
        # Please, do not change these parameters
        self.time_step = 32 # ms
        self.time_experiment = 60 # s
        
        # Reward zone coordinates
        self.reward_zone = {
            'x': 0.37,
            'y': -0.02, 
            'z': -0.16,
            'threshold': 0.1  # Distance threshold to consider robot in reward zone
        }

        # Initial robot position - will be used to reset robot at start of each simulation
        self.initial_position = {
            'translation': [0.007, 0, 0.35],
            'rotation': [-0.5, 0.5, 0.5, 2.09]
        }
        
        # Initiate Supervisor Module
        self.supervisor = Supervisor()
        # Check if the robot node exists in the current world file
        self.robot_node = self.supervisor.getFromDef("Controller")
        if self.robot_node is None:
            sys.stderr.write("No DEF Controller node found in the current world file\n")
            sys.exit(1)
        # Get the robots translation and rotation current parameters    
        self.trans_field = self.robot_node.getField("translation")  
        self.rot_field = self.robot_node.getField("rotation")
        
        # Check Receiver and Emitter are enabled
        self.emitter = self.supervisor.getDevice("emitter")
        self.receiver = self.supervisor.getDevice("receiver")
        self.receiver.enable(self.time_step)
        
        # Initialize the receiver and emitter data to null
        self.receivedData = "" 
        self.receivedWeights = "" 
        self.receivedFitness = "" 
        self.emitterData = ""
        
        ###########
        ### DEFINE here the 3 GA Parameters:
        self.num_generations = 50
        self.num_population = 20
        self.num_elite = 4
        
        # size of the genotype variable
        self.num_weights = 0
        
        # Creating the initial population
        self.population = []
        
        # All Genotypes
        self.genotypes = []
        
        # Display: screen to plot the fitness values of the best individual and the average of the entire population
        self.display = self.supervisor.getDevice("display")
        self.width = self.display.getWidth()
        self.height = self.display.getHeight()
        self.prev_best_fitness = 0.0;
        self.prev_average_fitness = 0.0;
        self.display.drawText("Fitness (Best - Red)", 0,0)
        self.display.drawText("Fitness (Average - Green)", 0,10)
        
        # Spotlight
        self.spotlight = self.supervisor.getFromDef('spotlight')  # Use the correct DEF name for the spotlight
        if self.spotlight is None:
            sys.stderr.write("No DEF SPOTLIGHT node found in the current world file\n")
            sys.exit(1)
        self.spotlight_on = False  # Initial state of the spotlight
        self.blink_interval = 10  # Number of steps between toggles
        self.blink_counter = 0  # Counter to track steps
        
    def createRandomPopulation(self):
        # Wait until the supervisor receives the size of the genotypes (number of weights)
        if(self.num_weights > 0):
            #  Size of the population and genotype
            pop_size = (self.num_population,self.num_weights)
            # Create the initial population with random weights
            self.population = np.random.uniform(low=-1.0, high=1.0, size=pop_size)

    def handle_receiver(self):
        while(self.receiver.getQueueLength() > 0):
            #Webots 2022: 
            #self.receivedData = self.receiver.getData().decode("utf-8")
            #Webots 2023: 
            self.receivedData = self.receiver.getString()
            typeMessage = self.receivedData[0:7]
            # Check Message 
            if(typeMessage == "weights"):
                self.receivedWeights = self.receivedData[9:len(self.receivedData)] 
                self.num_weights = int(self.receivedWeights)
            elif(typeMessage == "fitness"):  
                self.receivedFitness = float(self.receivedData[9:len(self.receivedData)])
            self.receiver.nextPacket()
        
    def handle_emitter(self):
        if(self.num_weights > 0):
            # Send genotype of an individual
            string_message = str(self.emitterData)
            string_message = string_message.encode("utf-8")
            #print("Supervisor send:", string_message)
            self.emitter.send(string_message)     
        
    def run_seconds(self,seconds):
        #print("Run Simulation")
        stop = int((seconds*1000)/self.time_step)
        iterations = 0
        while self.supervisor.step(self.time_step) != -1:
            self.handle_emitter()
            self.handle_receiver()
            if(stop == iterations):
                break    
            iterations = iterations + 1

    def check_reward_zone(self):
        # Get current robot position
        robot_pos = self.trans_field.getSFVec3f()
        
        # Calculate distance to reward zone
        distance_to_reward = np.sqrt(
            (robot_pos[0] - self.reward_zone['x'])**2 +
            (robot_pos[1] - self.reward_zone['y'])**2 +
            (robot_pos[2] - self.reward_zone['z'])**2
        )
        
        # Return True if robot is within threshold distance of reward zone
        return distance_to_reward <= self.reward_zone['threshold']

    def evaluate_genotype(self,genotype,generation):
        # Here you can choose how many times the current individual will interact with both environments
        # At each interaction loop, one trial on each environment will be performed
        numberofInteractionLoops = 3
        currentInteraction = 0
        fitnessPerTrial = []
        while currentInteraction < numberofInteractionLoops:
            #######################################
            # TRIAL: TURN RIGHT
            #######################################
            # Send genotype to robot for evaluation
            self.emitterData = str(genotype)
            
            # Reset robot position and physics
            self.reset_robot_position()
            
            # Evaluation genotype 
            self.run_seconds(self.time_experiment)
        
            # Measure fitness
            fitness = self.receivedFitness
            
            # Check for Reward and add it to the fitness value here
            # Add your code here
            
            print("Fitness: {}".format(fitness))     
                        
            # Add fitness value to the vector
            fitnessPerTrial.append(fitness)
            
            #######################################
            # TRIAL: TURN LEFT
            #######################################
            # Send genotype to robot for evaluation
            self.emitterData = str(genotype)
            
            # Reset robot position and physics
            self.reset_robot_position()
        
            # Evaluation genotype 
            self.run_seconds(self.time_experiment)
        
            # Measure fitness
            fitness = self.receivedFitness
            
            # Check for Reward and add it to the fitness value here
            # Add your code here
            
            print("Fitness: {}".format(fitness))
            
            # Add fitness value to the vector
            fitnessPerTrial.append(fitness)
            
            # End 
            currentInteraction += 1
            
        print(fitnessPerTrial)    
        
        fitness = np.mean(fitnessPerTrial)
        current = (generation,genotype,fitness)
        self.genotypes.append(current)  
        
        return fitness

    def run_demo(self):
        # Read File
        genotype = np.load("Best.npy")
        
        # Turn Left
        
        # Send Genotype to controller
        self.emitterData = str(genotype) 
        
        # Reset robot position and physics
        self.reset_robot_position()
    
        # Evaluation genotype 
        self.run_seconds(self.time_experiment) 
        
        # Measure fitness
        fitness = self.receivedFitness
        print("Fitness without reward or penalty: {}".format(fitness))
        
        # Turn Right
        
        # Send Genotype to controller
        self.emitterData = str(genotype) 
        
        # Reset robot position and physics
        self.reset_robot_position()
    
        # Evaluation genotype 
        self.run_seconds(self.time_experiment)  
        
        # Measure fitness
        fitness = self.receivedFitness
        print("Fitness without reward or penalty: {}".format(fitness))    
    
    def run_optimization(self):
        # Wait until the number of weights is updated
        while(self.num_weights == 0):
            self.handle_receiver()
            self.createRandomPopulation()
        
        print(">>>Starting Evolution using GA optimization ...\n")
        
        # For each Generation
        for generation in range(self.num_generations):
            print("Generation: {}".format(generation))
            current_population = []   
            # Select each Genotype or Individual
            for population in range(self.num_population):
                genotype = self.population[population]
                # Evaluate
                fitness = self.evaluate_genotype(genotype,generation)
                #print(fitness)
                # Save its fitness value
                current_population.append((genotype,float(fitness)))
                #print(current_population)
                
            # After checking the fitness value of all indivuals
            # Save genotype of the best individual
            best = ga.getBestGenotype(current_population);
            average = ga.getAverageGenotype(current_population);
            np.save("Best.npy",best[0])
            self.plot_fitness(generation, best[1], average);
            
            # Generate the new population using genetic operators
            if (generation < self.num_generations - 1):
                self.population = ga.population_reproduce(current_population,self.num_elite);
        
        #print("All Genotypes: {}".format(self.genotypes))
        print("GA optimization terminated.\n")   
    
    
    def draw_scaled_line(self, generation, y1, y2): 
        # the scale of the fitness plot
        XSCALE = int(self.width/self.num_generations);
        YSCALE = 100;
        self.display.drawLine((generation-1)*XSCALE, self.height-int(y1*YSCALE), generation*XSCALE, self.height-int(y2*YSCALE));
    
    def plot_fitness(self, generation, best_fitness, average_fitness):
        if (generation > 0):
            self.display.setColor(0xff0000);  # red
            self.draw_scaled_line(generation, self.prev_best_fitness, best_fitness);
    
            self.display.setColor(0x00ff00);  # green
            self.draw_scaled_line(generation, self.prev_average_fitness, average_fitness);
    
        self.prev_best_fitness = best_fitness;
        self.prev_average_fitness = average_fitness;
  
    def reset_robot_position(self):
        self.trans_field.setSFVec3f(self.initial_position['translation'])
        self.rot_field.setSFRotation(self.initial_position['rotation'])
        self.robot_node.resetPhysics()

    def toggle_spotlight(self):
        # Toggle the spotlight on and off
        self.spotlight_on = not self.spotlight_on
        self.spotlight.getField('on').setSFBool(self.spotlight_on)

    def send_robot_position(self):
        # Get the robot's current position
        robot_pos = self.trans_field.getSFVec3f()
        # Send the position as a string
        position_message = f"position: {robot_pos[0]}, {robot_pos[1]}, {robot_pos[2]}"
        self.emitter.send(position_message.encode("utf-8"))

if __name__ == "__main__":
    # Call Supervisor function to initiate the supervisor module   
    gaModel = SupervisorGA()
    
    # Function used to run the best individual or the GA
    keyboard = Keyboard()
    keyboard.enable(50)
    
    # Interface
    print("***************************************************************************************************")
    print("To start the simulation please click anywhere in the SIMULATION WINDOW(3D Window) and press either:")
    print("(S|s)to Search for New Best Individual OR (R|r) to Run Best Individual")
    print("***************************************************************************************************")
    
    while gaModel.supervisor.step(gaModel.time_step) != -1:
        # Blink the spotlight
        gaModel.blink_counter += 1
        if gaModel.blink_counter >= gaModel.blink_interval:
            gaModel.toggle_spotlight()
            gaModel.blink_counter = 0

        resp = keyboard.getKey()
        if(resp == 83 or resp == 65619):
            gaModel.run_optimization()
            print("(S|s)to Search for New Best Individual OR (R|r) to Run Best Individual")
            #print("(R|r)un Best Individual or (S|s)earch for New Best Individual:")
        elif(resp == 82 or resp == 65619):
            gaModel.run_demo()
            print("(S|s)to Search for New Best Individual OR (R|r) to Run Best Individual")
            #print("(R|r)un Best Individual or (S|s)earch for New Best Individual:")
