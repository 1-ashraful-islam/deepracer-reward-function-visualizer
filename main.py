import math
import matplotlib
# matplotlib.use('TkAgg')  # Set the backend to TkAgg
matplotlib.use('Qt5Agg') # Set the backend to Qt5Agg
#TODO: matplotlib is very slow in apple silicon for some reason. 


import matplotlib.pyplot as plt
import random
import numpy as np
import time
import matplotlib.animation as animation

#local imports
import util

# define constants
SPEED_MAX = 1.0
SPEED_MIN = 0.5
VELOCITY_MAX = SPEED_MAX # maximum velocity of the car in m/s
VELOCITY_MIN = SPEED_MIN # minimum velocity of the car in m/s
STEERING_ANGLE_MAX = 30.0
STEERING_ANGLE_MIN = -30.0
MAX_YAW_ANGLE_PER_STEP = 15.0
MAX_SPEED_CHANGE_PER_STEP = 0.5
DISCOUNT_FACTOR = 0.99


# Initialize the dictionary with default values for each parameter
params = {
    "all_wheels_on_track": True,        # Flag to indicate if the agent is on the track
    "x": 0.0,                            # Agent's x-coordinate in meters
    "y": 0.0,                            # Agent's y-coordinate in meters
    "closest_waypoints": [0, 1],         # Indices of the two nearest waypoints.
    "distance_from_center": 0.0,         # Distance in meters from the track center 
    "is_left_of_center": False,          # Flag to indicate if the agent is on the left side to the track center or not.
    "is_offtrack": False,                # Boolean flag to indicate whether the agent has gone off track.
    "is_reversed": False,                # Flag to indicate if the agent is driving clockwise (True) or counter clockwise (False).
    "heading": 0.0,                      # Agent's yaw in degrees
    "progress": 0.0,                     # Percentage of track completed
    "speed": 0.0,                        # Agent's speed in meters per second (m/s)
    "steering_angle": 0.0,               # Agent's steering angle in degrees
    "steps": 0,                          # Number steps completed
    "track_length": 0.0,                 # Track length in meters.
    "track_width": 0.0,                  # Width of the track
    "waypoints": [],                      # List of (x,y) as milestones along the track center
    "distance_between_traveled_waypoints": 0.0, # custom parameter to keep track of total distance in progress
    "best_closest_waypoints": [0, 1],    # custom parameter to keep track of the best closest waypoints (to detect when the car is going backwards)
    "is_on_center_line": True,          # custom parameter to keep track of if the car is on the center line
    "velocity": 0.0,                     # custom parameter to keep track of the velocity
    "reward_function_exploration": True,  # custom parameter to keep track of if the reward function is being visualized
}

#function to reset the parameters when the episode is done
def reset_params():
    #reset the parameters
    new_params = {
        "all_wheels_on_track": True,
        "x": 0.0,
        "y": 0.0,
        "closest_waypoints": [0, 1],
        "distance_from_center": 0.0,
        "is_left_of_center": False,
        "is_offtrack": False,
        "is_reversed": False,
        "heading": 0.0,
        "progress": 0.0,
        "speed": 0.0,
        "steering_angle": 0.0,
        "steps": 0,
        "distance_between_traveled_waypoints": 0.0,
        "best_closest_waypoints": [0, 1],
        "is_on_center_line": True,
        "velocity": 0.0,
    }
    update_params(new_params)


# function to update the parameters
def update_params(new_params):
    for key in new_params:
        if key in params:
            params[key] = new_params[key]

# function to access the parameters
def get_param(key):
    if key in params:
        return params[key]
    else:
        return None


# define a function for updating the closest waypoints based on the current position of the car
def get_closest_waypoints(obs):
    # get the parameters
    x, y = obs['x'], obs['y']
    waypoints = obs['waypoints']
    closest_waypoints = obs['closest_waypoints']
    best_closest_waypoints = obs['best_closest_waypoints']
    heading = obs['heading']
    track_length = obs['track_length']
    
    # get the current position of the car
    current_position = np.array([x,y])
    # get the closest waypoints
    closest_waypoints = closest_waypoints
    

    #lower bound of the waypoints to consider
    lower_bound = (closest_waypoints[0]-10)%len(waypoints)
    #upper bound of the waypoints to consider
    upper_bound = (closest_waypoints[1]+10)%len(waypoints)

    waypoints = np.array(waypoints)
    # get the relevant waypoints to check since lower_bound can be greater than upper_bound
    waypoints_tocheck = [waypoints [i % len (waypoints)] for i in range (lower_bound, upper_bound + len (waypoints))]
    
    # get the distances of the current position for 10 waypoints ahead and 10 waypoints behind of the closest waypoints
    # use np.linalg.norm to calculate the distance between the current position and the waypoints
    distances = np.linalg.norm(waypoints_tocheck - current_position, axis=1)
    # get the index of the closest waypoint
    closest_index = np.argmin(distances)

    #updated first closest waypoint
    first_closest_waypoint = (lower_bound+closest_index)%len(waypoints)
    #check if the closet_index is = closest_index + 1 since last waypoint is the same as the first waypoint
    if first_closest_waypoint == len(waypoints)-1:
        first_closest_waypoint = 0
    #updated second closest waypoint
    second_closest_waypoint = (first_closest_waypoint+1)%len(waypoints)

    # update the closest waypoints
    updated_closest_waypoints = [first_closest_waypoint, second_closest_waypoint]
    
    distance_traveled = 0
    # check if the closest waypoints are updated
    if updated_closest_waypoints[1] != closest_waypoints [1]:
        #figure out if the closest waypoints are updated in the forward or backward direction
        # get the coordinates of the previous, current and next waypoints
        prev_x, prev_y = waypoints[best_closest_waypoints[0]] # to detect going forward or backward from the best closest waypoints
        curr_x, curr_y = waypoints[best_closest_waypoints[1]]
        next_x, next_y = waypoints[updated_closest_waypoints[1]]

        # direction of the best closest waypoints
        dir = np.sign((next_x - curr_x) * (curr_y - prev_y) - (next_y - curr_y) * (curr_x - prev_x))

        start_idx = closest_waypoints[0]
        end_idx = updated_closest_waypoints[0]
        
        # calculate the distance traveled
        distance = np.sum([math.dist(waypoints [i % len (waypoints)], waypoints [(i+1) % len (waypoints)]) for i in range (start_idx, end_idx)])

        if distance == 0:
            # either going backward or the circularity of the waypoints causing the range to fail
            # check distance with circularity
            distance1 = np.sum([math.dist(waypoints [i % len (waypoints)], waypoints [(i+1) % len (waypoints)]) for i in range (start_idx, end_idx+ len(waypoints))]) #np.sum([math.dist(waypoints [i % len (waypoints)], waypoints [(i+1) % len (waypoints)]) for i in range (start_idx, end_idx + len(waypoints))])
            distance2 = track_length - distance1
            if distance1 < distance2: # circularity was causing the range to fail
                distance = distance1
                
        if distance == 0: # if distance is still 0, then it is going backward
            #going backward
            start_idx = updated_closest_waypoints[0]
            end_idx = closest_waypoints[0]
            distance1 = np.sum([math.dist(waypoints [i % len (waypoints)], waypoints [(i+1) % len (waypoints)]) for i in range (start_idx, end_idx)])
            if distance1 ==  0:
                distance1 = np.sum([math.dist(waypoints [i % len (waypoints)], waypoints [(i+1) % len (waypoints)]) for i in range (start_idx, end_idx+ len(waypoints))]) #np.sum([math.dist(waypoints [i % len (waypoints)], waypoints [(i+1) % len (waypoints)]) for i in range (start_idx, end_idx + len(waypoints))])
                distance2 = track_length - distance1
                if distance1 < distance2: # circularity was causing the range to fail
                    distance = distance1
                    
            distance = - distance1

        # do a min max to make sure distance is not too high or too low
        distance = max(min(distance, track_length - distance), - track_length - distance)

        if abs(distance) > 2*SPEED_MAX:
            print("Something is wrong! Check the distances! distance", distance, "closest waypoints: ", closest_waypoints, "updated closest waypoints: ", updated_closest_waypoints)
        distance_traveled = distance

    
    # return the updated closest waypoints
    return updated_closest_waypoints, distance_traveled


# define a function that simulates the agent's action (speed, steering angle) and modifies the relevant parameters and calls the reward function
def simulate_action(obs,speed, steering_angle,rf):

   
    # speed change for this step
    speed_change = speed - obs["speed"]
    if speed_change >= 0:
        obs["velocity"]  = min(VELOCITY_MAX, obs["velocity"] + min(MAX_SPEED_CHANGE_PER_STEP, speed_change))
    else:
        obs["velocity"]  = max(VELOCITY_MIN,obs["velocity"] - min(MAX_SPEED_CHANGE_PER_STEP, -speed_change))

     # update speed
    obs["speed"] = speed
    
    # update heading by steering angle/5 # changed from 15
    
    new_heading = math.fmod(obs["heading"] + math.copysign(min(MAX_YAW_ANGLE_PER_STEP, abs(steering_angle)),steering_angle),360.0)

    # force it to be the positive remainder, so that 0 <= angle < 360
    new_heading = (new_heading + 180.0) % 360.0 - 180.0

    obs["heading"] = new_heading


    # update x and y coordinates
    obs["x"] = obs["x"] +  obs["velocity"]  * math.cos(math.radians(obs["heading"]))/15.0
    obs["y"] = obs["y"] +  obs["velocity"]  * math.sin(math.radians(obs["heading"]))/15.0

    # update the closest waypoints
    closest_waypoints,distance_traveled = get_closest_waypoints(obs)

    # update the distance traveled
    obs["distance_between_traveled_waypoints"] += distance_traveled
    # update the closest waypoints
    obs["closest_waypoints"] = closest_waypoints
    # if the distance_traveled is positive, the car is moving forward, update the best closest waypoints
    if distance_traveled > 0:
        obs["best_closest_waypoints"] = closest_waypoints

    # closest waypoints
    # closest_waypoints = obs["closest_waypoints"]
    first_point = obs["waypoints"][closest_waypoints[0]]
    second_point = obs["waypoints"][closest_waypoints[1]]

    # update progress
    projected_distance_from_first_waypoint = util.projected_distance(first_point, second_point, [obs["x"], obs["y"]])
    obs["progress"] = 100.0*(projected_distance_from_first_waypoint + obs["distance_between_traveled_waypoints"]) / obs["track_length"]

    # distance from center is the vertical distance from the line connecting the closest waypoints
    obs["distance_from_center"] = util.vertical_distance_from_two_points((obs["x"],obs["y"]),first_point,second_point)
    # check if the car is off track
    if obs["distance_from_center"] > (0.5*obs["track_width"] + 0.075): # assuming the car is 15cm wide
        obs["is_offtrack"] = True
    else:
        obs["is_offtrack"] = False

    if obs["distance_from_center"] >= 0.5*obs["track_width"]:
        obs["all_wheels_on_track"] = False
    else:
        obs["all_wheels_on_track"] = True
        # print("off track")

    # calculate the cross product of (p2 - p1) and (point - p1)
    # if the result is positive, the point is on the left side of the line
    # if the result is negative, the point is on the right side of the line
    # if the result is zero, the point is on the line
    # p2 is the second waypoint, p1 is the first waypoint, point is the car
    cross_product = (second_point[0] - first_point[0]) * (obs["y"] - first_point[1]) - (second_point[1] - first_point[1]) * (obs["x"] - first_point[0])

    # update is_left_of_center
    if cross_product > 0:
        obs["is_left_of_center"] = True
        obs["is_on_center_line"] = False
    elif cross_product < 0:
        obs["is_left_of_center"] = False
        obs["is_on_center_line"] = False
    else:
        obs["is_on_center_line"] = True

    # update steering angle
    obs["steering_angle"] = steering_angle
    # update steps
    obs["steps"] += 1
    

    #call the reward function
    reward = rf(obs)
    return obs,reward

# function to load the track
def load_track(track_name):
    # define a dictionary to store the track data
    track = {}
    # load the track data from the same directory as the notebook
    track_data = np.load('tracks/'+track_name+'.npy')
    # segment track_data into outer, inner and center
    # set first and second column as x and y coordinates of waypoints
    
    waypoints = track_data[:,0:2]
    outer_border = track_data[:,4:6]
    inner_border = track_data[:,2:4]

    # calculate track_length by summing the euclidean distance between each waypoint
    track_length = 0
    for i in range(len(waypoints)-1):
        track_length += np.sqrt((waypoints[i+1,0]-waypoints[i,0])**2 + (waypoints[i+1,1]-waypoints[i,1])**2)
    # round the track_length to 2 decimals
    track_length = round(track_length, 2)

    # calculate the track_width by averaging the euclidean distance between each outer and inner border
    track_width = 0
    for i in range(len(outer_border)):
        track_width += np.sqrt((outer_border[i,0]-inner_border[i,0])**2 + (outer_border[i,1]-inner_border[i,1])**2)
    track_width /= len(outer_border)

    
    # round the track_width to 2 decimals
    track_width = round(track_width, 2)

    # store the track data in the track dictionary
    track["waypoints"] = waypoints
    track["outer_border"] = outer_border
    track["inner_border"] = inner_border
    track["track_length"] = track_length
    track["track_width"] = track_width
    return track

# define function to get a action sample
def get_action_sample():
    # random speed between 0.5 and 1.0
    speed = random.uniform(SPEED_MIN, SPEED_MAX)

    # random steering angle between -30 and 30
    steering_angle = random.uniform(STEERING_ANGLE_MIN, STEERING_ANGLE_MAX)

    return speed, steering_angle

# define a function that takes an action and returns the next observation
def get_next_step(params, rf, max_long_term_depth):

    # get action for the agent
    speed, steering_angle = get_action_sample()
    
    # simulate the action, get the short term reward
    obs,short_term_reward = simulate_action(dict(params), speed, steering_angle, rf)


    ## TODO: big limitation! if the reward function has any state (global variables), it will keep getting updated while checking for the best action
    ## this causes unintended behavior of the reward function

    # initialize the long term reward to short term reward
    long_term_reward = short_term_reward
    # initialize the discount to DISCOUNT_FACTOR and update by DISCOUNT_FACTOR every future step
    discount = DISCOUNT_FACTOR
    # initialize obs for the long term reward
    long_term_obs = dict(obs)
    
    for k in range(max_long_term_depth):
        # get action for the agent
        speed, steering_angle = get_action_sample()
        # simulate the action, get the short term reward
        long_term_obs,action_reward = simulate_action(long_term_obs, speed, steering_angle, rf)
        # use the discount factor to calculate the long term reward
        long_term_reward = long_term_reward + discount*action_reward
        # update the discount
        discount = discount*DISCOUNT_FACTOR
        # if the car is off the track then break
        if long_term_obs["is_offtrack"] == True:
            break
    
    
    return obs, short_term_reward, long_term_reward


# define a main function
def main(track_name,rf, save_animation=False,max_action_per_step=10, max_long_term_depth=5, max_steps=1000, max_episode=10,animation_name="animation", show_exploration= False):
    # load the track
    track = load_track(track_name)
    # update the track parameters
    track_params = {
    "track_length": track["track_length"],
    "track_width": track["track_width"],
    "waypoints": track["waypoints"],
    }
    update_params(track_params)
    # figure out the size of the figure
    x_fig_size = abs(np.max(track["outer_border"][:,0],axis=0) - np.min(track["outer_border"][:,0],axis=0))
    y_fig_size = abs(np.max(track["outer_border"][:,1],axis=0) - np.min(track["outer_border"][:,1],axis=0))
    aspect_ratio = x_fig_size/y_fig_size
    # create a figure
    fig = plt.figure(figsize=(10,10/aspect_ratio))

    # create a video writer
    Writer = animation.FFMpegWriter(fps=15, extra_args=['-vcodec', 'libx264']) #metadata=dict(artist='Me'), bitrate=1800)
    if save_animation:
        Writer.setup(fig, animation_name+'.mp4', 100)
    
    # turn on interactive mode
    plt.ion() # set plot to animated
    # plot the track
    plt.plot(track["waypoints"][:,0], track["waypoints"][:,1], color="green",alpha=0.5)
    plt.plot(track["outer_border"][:,0], track["outer_border"][:,1], color="blue",alpha=0.5)
    plt.plot(track["inner_border"][:,0], track["inner_border"][:,1], color="blue",alpha=0.5)
    # mark the start and finish line
    plt.plot(track["waypoints"][0,0], track["waypoints"][0,1], marker="o", color="red",alpha=0.5)
    
    plt.draw()
    plt.pause(0.0001)
    # Add a variable for colorbar to the plot and save its axes
    cax = None
    if save_animation:
        Writer.grab_frame()
    
    
    # collect stats for all the episodes
    episode_num = []
    episode_reward = []
    episode_steps = []
    episode_time = []
    episode_progress = []
    episode_travelled_distance = []
    episode_avg_speed = []

    last_waypoint = 0
    
    for m in range(max_episode):
        
        reset_params()
        # starting waypoint
        start_waypoint = last_waypoint % len(params["waypoints"])
        start_x, start_y = params["waypoints"][start_waypoint]
        closest_waypoints = [start_waypoint, (start_waypoint+1)%len(params["waypoints"])]
        #start_heading = closest waypoint segment heading of the track
        start_heading = math.degrees(math.atan2(params["waypoints"][closest_waypoints[1]][1] - params["waypoints"][closest_waypoints[0]][1], params["waypoints"][closest_waypoints[1]][0] - params["waypoints"][closest_waypoints[0]][0]))

        #define initial conditions
        new_params = {
        "x": start_x,
        "y": start_y,
        "closest_waypoints": closest_waypoints,
        "heading": start_heading,
        "best_closest_waypoints": closest_waypoints,
        }
        update_params(new_params)

        
        # place holder list for x,y, and reward
        x_list = []
        y_list = []
        reward_list = []
        speed_list = []
        velocity_list = []

        
        #additional termination crtieria
        # if the agent is going backwards for more than 5 steps
        backward_steps = 0
        max_distance_travelled = 0.0

        #simulate the agent's action in a for loop and plot the x,y coordinates and the reward
        for i in range(max_steps):
            best_obs = {}
            best_long_term_reward = -1000.0
            best_short_term_reward = 0.0    
            
            # simulate the action, get the short term reward
            #TODO: streamline this option so that the states don't get updated during exploration
            # As a work around, make sure the reward function is called with the exploration flag set to True to avoid updating the states of the reward function
            # TODO: if you are using the work around, make sure to include this logic in your reward function
            params["reward_function_exploration"] = True

            # create a copy of the params to use for the exploration
            params_copy = dict(params)

            for j in range(max_action_per_step):

                # # simulate the action, get the short term reward

                obs, short_term_reward, long_term_reward = get_next_step(params_copy,rf, max_long_term_depth)

                if show_exploration:
                    x_list.append(obs["x"])
                    y_list.append(obs["y"])
                    reward_list.append(short_term_reward)
                    speed_list.append(obs["speed"])
                    velocity_list.append(obs["velocity"])

                # ## TODO: big limitation! if the reward function has any state, it will keep getting updated while checking for the best action
                # ## this causes unintended behavior of the reward function

                # if best_obs empty the update with the first action
                if not best_obs:
                    best_long_term_reward = long_term_reward
                    best_obs = obs
                    best_short_term_reward = short_term_reward
                if (long_term_reward > best_long_term_reward and random.random() > 0.2) or (long_term_reward == best_long_term_reward and random.random() > 0.5): # add randomness 
                    best_long_term_reward = long_term_reward
                    best_obs = obs
                    best_short_term_reward = short_term_reward
                    
                
            
            
            
            if abs(best_obs["progress"] - params["progress"])  > 5.0:
                print("Ridiculous progress .prev -> next", params["progress"], best_obs["progress"])
            if (best_obs["progress"] >  (0.5 + params["progress"])) and (best_short_term_reward <= 1e-3) and (best_obs["is_offtrack"] == False):
                print("progress diff: {:.2f}".format(best_obs["progress"] - params["progress"]) ,"but reward: {:.3f}".format( best_short_term_reward))

            
            # call the reward function to get the reward and
            # turn the reward_function_exploration to false
            # this is to make sure any state in the reward function is only updated with the best_obs
            # TODO: big limitation! if the reward function has any state, it will not update with modification, which may cause the long term reward to be incorrect
            # this causes unintended behavior of the reward function, but slightly better than updating the reward function states with all the exploration actions
            best_obs["reward_function_exploration"] = False
            best_reward = rf(best_obs)
            best_obs["reward_function_exploration"] = True # reset the reward_function_exploration to True

            # update the params with the best_obs
            update_params(best_obs)
            # check if the car is going backwards 0.5m is arbitrarily selected threshold to reduce false positive
            if best_obs["distance_between_traveled_waypoints"] > max_distance_travelled:
                max_distance_travelled = best_obs["distance_between_traveled_waypoints"]
                backward_steps = 0
            elif best_obs["distance_between_traveled_waypoints"] < max_distance_travelled - 0.25:
                backward_steps += 1

            
            # this may only happen if our closest_waypoints are messed up
            if params["closest_waypoints"][0] == params["closest_waypoints"][1]:
                print("whew! we messed up")
                time.sleep(10)
            

            # append the x,y,reward, speed to the list
            if not show_exploration:
                x_list.append(params["x"])
                y_list.append(params["y"])
                reward_list.append(best_reward)
                speed_list.append(params["speed"])
                velocity_list.append(best_obs["velocity"])

            # terminate the epsiode if the car is off the track or the progress is 100% or the car is going backwards for more than 5 steps
            if params["is_offtrack"] or params["progress"] >= 100 or backward_steps > 15:
                break


        # make sure reward_list values are suitable to use as size
        # scaling factor
        max_speed = np.max(speed_list)
        min_speed = np.min(speed_list)
        scale_factor = 1.0/(max_speed - min_speed)
        scaling_list = [((x - min_speed)*scale_factor)**3 for x in speed_list]

        #TODO Check if theres any ridiculously large or small reward and discard those. 
        # maybe use the mean and standard deviation to discard outliers
        # make sure reward_list values are suitable to use as color
        color_scaling_factor = 1.0 /(np.mean(reward_list) + 3*np.std(reward_list))
        color_list = [min(1.0,max(0,x*color_scaling_factor)) for x in reward_list]

        
        #plot scatter x,y coordinates with reward as size and color
        # TODO: parameterize the plotRewardAsColor instead of hardcoding
        plotRewardAsColor = False
        if plotRewardAsColor:
            plt.scatter(x_list, y_list, s=scaling_list, c=color_list, cmap='jet', alpha=0.5)
        else:
            plt.scatter(x_list, y_list, s=10, c=speed_list, cmap='jet', alpha=0.5)

        if cax == None:
            cax = plt.colorbar().ax
        else:
            # Update the colorbar using the same axes
            plt.colorbar(cax=cax)
        if params["is_offtrack"]:
            plt.plot(x_list[-1], y_list[-1], 'x',color=[0,0,0,0.5])
        
        plt.draw()
        plt.pause(0.0001)
        if save_animation:
            Writer.grab_frame()

        # # plot the rewards in a separate plot
        # calculate total distance travelled using the x,y coordinates
        total_distance = 0
        for i in range(len(x_list)-1):
            total_distance += np.sqrt((x_list[i+1]-x_list[i])**2 + (y_list[i+1]-y_list[i])**2)


        # update episode stats for completed ones to avoid skewing the plots
        if params["progress"] >= 100 or params["steps"] >= max_steps:
            episode_num.append(m)
            episode_reward.append(np.sum(reward_list))
            episode_progress.append(params["progress"])
            episode_steps.append(params["steps"])
            episode_avg_speed.append(np.mean(speed_list))
            episode_time.append((params["steps"]/15.0)*(100.0/params["progress"]))
            episode_travelled_distance.append(total_distance)

        # print episode stats
        print(
                "e: {}".format(m),
                "center distance: {:.2f}".format(params["distance_from_center"]),
                "progress: {:.2f}".format(params["progress"]),
                "steps: {}".format(params["steps"]),
                "reward: {:.3f}".format(np.sum(reward_list)),
                "mean speed: {:.2f}".format(np.mean(speed_list)),
                "mean velocity: {:.2f}".format(np.mean(velocity_list)),
                "reward/step + std: [{:.3f} {:.3f}]".format(np.sum(reward_list)/params["steps"] , np.std(reward_list)),
                "progress/step: {:.3f}".format(params["progress"]/params["steps"]),
                "lap time: {:.2f} s".format((params["steps"]/15.0)*(100.0/params["progress"])),
                "total distance: {:.2f} m".format(total_distance),
                )

        last_waypoint = params["closest_waypoints"][0] + random.randint(0,50)
            
    print("done")
    # # Add a colorbar to the plot
    
    # if save_animation is True, create a video of the scatter plots using animation
    if save_animation:
        # grab the last frame
        Writer.grab_frame()
        # save the animation as mp4 video file
        Writer.finish()
    # save the plot as png
    plt.savefig("{}.png".format("graph_latest"))
    plt.ioff() # turn off interactive mode
    # plot a second figure with subplots for episode stats
    fig2 = plt.figure(2, figsize=(10,8))
    # add title to the figure
    fig2.suptitle("Completed Episode stats")
    plt.subplot(2,2,1)
    plt.scatter(episode_steps, episode_reward)
    plt.xlabel("episode steps")
    plt.ylabel("reward")
    plt.subplot(2,2,2)
    plt.scatter(episode_steps, episode_travelled_distance)
    plt.xlabel("episode steps")
    plt.ylabel("travelled distance")
    plt.subplot(2,2,3)
    plt.scatter(episode_steps,episode_avg_speed)
    plt.xlabel("episode steps")
    plt.ylabel("avg speed")
    plt.subplot(2,2,4)
    plt.scatter(episode_time, episode_reward)
    plt.xlabel("episode time")
    plt.ylabel("episode reward")
    
    plt.savefig("{}.png".format("graph_latest_stats"))


    plt.show() # show the plot
    
    return
   
    

# define entry point
if __name__ == '__main__':
    import reward_function_examples as rf
    # import reward_function_current as rf
    track = "jyllandsringen_pro_cw" # 
    save_animation = False
    animation_name = "updated_follow_center_line_example"
    reward_function = rf.reward_function

    ## config for best path
    max_action_per_step = 15 # number of actions to try per step for short term reward
    max_long_term_depth = 10 # number of steps to look ahead for long term reward
    max_steps = 1200 # maximum number of steps per episode
    max_episode = 15 # maximum number of episodes to simulate
    show_exploration = False # show the all the exploration actions # recommended with less actions per step and depth to avoid slow down

    ## config for reward visualization in all track
    show_all_track = False
    if show_all_track:
        max_action_per_step = 1
        max_long_term_depth = 1
        max_steps = 1000
        max_episode = 100
        show_exploration = False

    main(track, reward_function, save_animation, max_action_per_step, max_long_term_depth, max_steps, max_episode, animation_name, show_exploration)

