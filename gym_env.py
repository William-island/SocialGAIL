import gym
import pygame
import numpy as np
import os
import imageio
import pickle
import argparse
import random
import math
import os
from statistics import mean
from scipy.spatial.distance import euclidean
from GNN_models.graph_data import GraphData
import torch

class CrowdEnv():  # can extend from gym.Env
    def __init__(self, args):
        super(CrowdEnv, self).__init__()

        self.time_interval = args.time_interval
        self.trajectorys, self.frame_data, self.start_data, self.goal_data = self._load_dataset(args.dataset_path)
        self.regions = args.regions
        self.radius = args.radius
        self.num_agents = len(self.trajectorys.keys())
        self.frame_interval = args.frame_interval
        self.map_size_bound = args.map_size_bound # [low_x, high_x, low_y, high_y]
        self.with_last_speed = args.with_last_speed
        self.observation_type = args.observation_type # 'radar' / 'graph'
        if self.observation_type == 'graph':
            self.graph_obs_past_len = args.graph_obs_past_len
            self.padd_to_number = args.padd_to_number
            self.graph_feature_len = args.grahp_feature_len

        self.agent_id = 1
        self.frame_number = 0
        self.current_position = [self.trajectorys[self.agent_id][0][0], self.trajectorys[self.agent_id][0][1]]
        self.target_circle_with = 1
        self.time_steps = 0
        self.end_frame = 120000
        self.ADE_list = []
        self.new_traj = [self.current_position]
        self.old_traj = np.array(self.trajectorys[self.agent_id])[:,0:2]

        ## gym action space & observation sapce
        # self.action_space = gym.spaces.Box(low=-10, high=10, shape=(2,), dtype=np.float32)
        # if self.with_last_speed:
        #     self.observation_space = gym.spaces.Box(low=-400, high=400, shape=(3*self.regions+2+2,), dtype=np.float64)
        # else:
        #     self.observation_space = gym.spaces.Box(low=-400, high=400, shape=(3*self.regions+2,), dtype=np.float64)

    def reset(self):
        # decide the agent of current eposide and start frame
        self.agent_id = random.randint(1,self.num_agents)
        # self.agent_id = 3408
        while(not self._frame_continues_check()):
            self.agent_id = random.randint(1,self.num_agents)
        # print("now agent_id{}".format(self.agent_id))
        self.frame_number = self.trajectorys[self.agent_id][0][2]
        self.current_position = np.array([self.trajectorys[self.agent_id][0][0], self.trajectorys[self.agent_id][0][1]])
        self.time_steps = 0
        self.end_frame = self.trajectorys[self.agent_id][-1][2]
        self.ADE_list = []
        self.new_traj = [self.current_position]
        self.old_traj = np.array(self.trajectorys[self.agent_id])[:,0:2]
        return self._get_observation()

    def step(self, action):
        # update frame
        self._update_frame_number()

        # update the position of target agent
        # valid check: map & collision
        reward = 0
        done = False
        self.time_steps += 1

        # update new position , get position change via speed!!
        # dx = action[0]
        # dy = action[1]
        dx = action[0]*self.time_interval
        dy = action[1]*self.time_interval
        new_position = np.array(self.current_position) + np.array([dx, dy])

        # check map validation
        if not self._valid_check_in_map():
            new_position = self.current_position

        # check collision 
        for other, opos in self.frame_data[self.frame_number].items():
            if self.agent_id != other:
                dist = np.linalg.norm(new_position - np.array([opos[0],opos[1]]))
                if dist < 1.0:
                    ## handle collision：back to the old position and punish it
                    new_position = self.current_position # no step back
                    reward -= 1
        self.current_position = new_position

        self.new_traj.append(self.current_position)

        # reward each step
        # gt_x = self.frame_data[self.frame_number][self.agent_id][0]
        # gt_y = self.frame_data[self.frame_number][self.agent_id][1]
        # step_dist = np.linalg.norm(np.array(self.current_position) - np.array(gt_x, gt_y))
        # reward -= step_dist

        # end condition 1
        dist = np.linalg.norm(np.array(self.current_position) - np.array(self.goal_data[self.agent_id]))
        self.ADE_list.append(dist)
        if dist < self.target_circle_with:
            # to the target circle
                reward += 10
                done = True
        # end condition2
        if self.frame_number == self.end_frame:
            done = True
        # end condition 3
        if self.time_steps > 400:
            done = True

        return self._get_observation(), reward, done, {}
    

    def render(self):
        # 创建 Pygame Surface 对象
        x_shift, y_shift = 12, 15
        scale = 10
        screen_width, screen_height = 60*scale, 60*scale
        screen = pygame.Surface((screen_width, screen_height))

        screen.fill((255, 255, 255))  # clear Surface

        for waypoint in self.trajectorys[self.agent_id]:
            pygame.draw.circle(screen, (185, 199, 141), (int((waypoint[0]+x_shift)*scale), int((waypoint[1]+y_shift)*scale)), 0.5*scale)

        for person,value in self.frame_data[self.frame_number].items():
            # draw the agent
            if person != self.agent_id:
                agent_x, agent_y = value[0]+x_shift, value[1]+y_shift
                pygame.draw.circle(screen, (62, 115, 158), (int(agent_x*scale), int(agent_y*scale)), 0.5*scale)
            else:
                pygame.draw.circle(screen, (255, 0, 0), (int((self.current_position[0]+x_shift)*scale), int((self.current_position[1]+y_shift)*scale)), 0.5*scale)
                pygame.draw.circle(screen, (255, 0, 0), (int((self.goal_data[self.agent_id][0]+x_shift)*scale), int((self.goal_data[self.agent_id][1]+y_shift)*scale)), 1*scale, 2)
        

        # 保存为图片文件
        pygame.image.save(screen, f'./gym_render/temp_pic/{int(self.frame_number/self.frame_interval)}.png')
    
    



    def generate_gif(self):
        # draw gif
        pic_list = sorted(os.listdir('./gym_render/temp_pic/'),key = lambda x:int(x[:-4]))
        with imageio.get_writer(uri=f'./gym_render/id_{self.agent_id}.gif', mode='i', fps=15) as writer:
            for pic in pic_list:
                writer.append_data(imageio.imread('./gym_render/temp_pic/'+pic))
        # delete all pictures
        for pic in pic_list:
            os.remove('./gym_render/temp_pic/'+pic)

    def compute_ADE(self):
        ade = mean(self.ADE_list)
        return ade
    
    def compute_FDE(self):
        fde = euclidean(self.current_position, self.old_traj[-1])
        return fde
    
    def compute_Frechet_Distance(self):
        P = self.old_traj
        Q = self.new_traj
        p_length = len(P)
        q_length = len(Q)
        distance_matrix = np.ones((p_length, q_length)) * -1

        # fill the first value with the distance between
        # the first two points in P and Q
        distance_matrix[0, 0] = euclidean(P[0], Q[0])

        # load the first column and first row with distances (memorize)
        for i in range(1, p_length):
            distance_matrix[i, 0] = max(distance_matrix[i - 1, 0], euclidean(P[i], Q[0]))
        for j in range(1, q_length):
            distance_matrix[0, j] = max(distance_matrix[0, j - 1], euclidean(P[0], Q[j]))

        for i in range(1, p_length):
            for j in range(1, q_length):
                distance_matrix[i, j] = max(
                    min(distance_matrix[i - 1, j], distance_matrix[i, j - 1], distance_matrix[i - 1, j - 1]),
                    euclidean(P[i], Q[j]))
        # distance_matrix[p_length - 1, q_length - 1]
        return distance_matrix[p_length-1,q_length-1]
    
    def compute_DTW(self):
        s1 = self.old_traj
        s2 = self.new_traj
        m = len(s1)
        n = len(s2)

        # 构建二位dp矩阵,存储对应每个子问题的最小距离
        dp = [[0]*n for _ in range(m)] 

        # 起始条件,计算单个字符与一个序列的距离
        for i in range(m):
            dp[i][0] = euclidean(s1[i],s2[0])
        for j in range(n):
            dp[0][j] = euclidean(s1[0],s2[j])
        
        # 利用递推公式,计算每个子问题的最小距离,矩阵最右下角的元素即位最终两个序列的最小值
        for i in range(1,m):
            for j in range(1,n):
                dp[i][j] = min(dp[i-1][j-1],dp[i-1][j],dp[i][j-1]) + euclidean(s1[i],s2[j])
        
        return dp[-1][-1]




    def _valid_check_in_map(self):
        valid = True

        # get parameters
        current_x = self.current_position[0]
        current_y = self.current_position[1]
        low_x = self.map_size_bound[0]
        high_x = self.map_size_bound[1]
        low_y = self.map_size_bound[2]
        high_y = self.map_size_bound[3]

        # bound validation check
        if (current_x<low_x) or (current_x>high_x) or (current_y<low_y) or (current_y>high_y):
            valid = False

        # grid map obstacle check
        if not self._gridmap_check:
            valid = False

        return valid
    

    def _load_gridmap(self):
        # init grid map
        grid_map = {}
        x_length = math.ceil(self.map_size_bound[1]-self.map_size_bound[0])
        y_length = math.ceil(self.map_size_bound[3]-self.map_size_bound[2])
        for i in range(x_length*y_length):
            grid_map[i] = True
        return grid_map
    
    # 还没实现！
    def _gridmap_check(self):
        return True

    def _load_dataset(self, dataset_path):
        # The dataset is organized as per person trajectory dict as form of pickle
        # fist load that then transfer to per frame form
        # trajectorys contains the trajectory of every pedestrian, each waypoint is [x,y,frame_number]
        with open(dataset_path, 'rb') as f:
            trajectorys = pickle.load(f)

        # delet pedestrians with only one or two waypoints
        for id in list(trajectorys.keys()):
            if len(trajectorys[id]) < 3:
                del trajectorys[id]
        print(f'dataset pedestrians:{len(trajectorys)}')

        # calculate last speed and next speed
        for traj in trajectorys.values():
            len_traj = len(traj)
            for i in range(1, len_traj - 1):
                past_v_x = (traj[i][0] - traj[i - 1][0]) / self.time_interval
                past_v_y = (traj[i][1] - traj[i - 1][1]) / self.time_interval
                future_v_x = (traj[i + 1][0] - traj[i][0]) / self.time_interval
                future_v_y = (traj[i + 1][1] - traj[i][1]) / self.time_interval
                traj[i].append(past_v_x)
                traj[i].append(past_v_y)
                traj[i].append(future_v_x)
                traj[i].append(future_v_y)
            traj[0].append(traj[1][3])
            traj[0].append(traj[1][4])
            traj[0].append(traj[1][3])
            traj[0].append(traj[1][4])
            traj[len_traj - 1].append(traj[len_traj - 2][5])
            traj[len_traj - 1].append(traj[len_traj - 2][6])
            traj[len_traj - 1].append(traj[len_traj - 2][5])
            traj[len_traj - 1].append(traj[len_traj - 2][6])

        # transfer to per frame dict
        # frame_data: key is frame id, value is all the pedestrains' waypoints
        frame_data = {}
        for id in list(trajectorys.keys()):
            for waypoint in trajectorys[id]:
                frame_data[waypoint[2]] = {}
        for id in list(trajectorys.keys()):
            for waypoint in trajectorys[id]:
                frame_data[waypoint[2]][id] = [waypoint[0], waypoint[1], \
                                               waypoint[3], waypoint[4], waypoint[5],waypoint[6]]

        # save each pedestrian's start position
        start_data={}
        for id in list(trajectorys.keys()):
            info=trajectorys[id][0]
            start_data[id]=[info[0],info[1]]

        # save each pedestrian's goal
        goal_data={}
        for id in list(trajectorys.keys()):
            info=trajectorys[id][-1]
            goal_data[id]=[info[0],info[1]]

        return trajectorys, frame_data, start_data, goal_data
    

    # Choose the type of observation
    def _get_observation(self):
        if self.observation_type=='radar':
            return self._get_radar_observation()
        if self.observation_type=='graph':
            return self._get_graph_observation()

    def _get_radar_observation(self):
        # get the observation of current state

        # init state
        state = [0 for i in range(3*self.regions+2)]

        for i in range(self.regions):
            state[i*3] = self.radius


        # get the direction to goal
        dx, dy = self.goal_data[self.agent_id][0]-self.current_position[0], self.goal_data[self.agent_id][1]-self.current_position[1]
        if dx!=0 or dy!=0: # 归一化
            vx, vy = dx/math.sqrt(dx**2+dy**2),dy/math.sqrt(dx**2+dy**2)
            state[-2], state[-1] = vx, vy
        else:
            state[-2], state[-1] = 0, 0

        # with last_speed
        if self.with_last_speed:
            if len(self.new_traj) == 1:
                last_v = [0, 0]
            else:
                last_v = (self.current_position - self.new_traj[-2])/self.time_interval
            state.extend(last_v)

        
        # get nearby information
        for other,ovalue in self.frame_data[self.frame_number].items():
            if other!= self.agent_id:
                x2=ovalue[0]
                y2=ovalue[1]
                vec=[x2-self.current_position[0],y2-self.current_position[1]]
                dis=math.sqrt((self.current_position[0]-x2)**2+(self.current_position[1]-y2)**2)
                ang=self._angle_by_x(vec)
                region_order=int((ang*self.regions)/360)
                if dis<self.radius:
                    if dis<state[3*region_order]:
                        state[3*region_order]=dis
                        state[3*region_order+1]=ovalue[2]
                        state[3*region_order+2]=ovalue[3]
        return np.array(state)

    def _get_graph_observation(self):
        X=[]
        cluster=[]
        edge_index=[[],[]]

        # goal vector
        dx, dy= self.goal_data[self.agent_id][0]-self.current_position[0], self.goal_data[self.agent_id][1]-self.current_position[1]
        if dx!=0 or dy!=0: # 归一化
            vx,vy=dx/math.sqrt(dx**2+dy**2),dy/math.sqrt(dx**2+dy**2)
            goal=[vx,vy]
        else:
            goal=[0,0]

        # 添加自身上一时刻速度
        if len(self.new_traj) == 1:
            last_speed = [0, 0]
        else:
            last_speed = (self.current_position - self.new_traj[-2])/self.time_interval

        # agent当前位置
        x1=self.current_position[0]
        y1=self.current_position[1]

        # Insert target person
        if len(self.new_traj) == 1:
            last_x = 0
            last_y = 0
        else:
            last_x = self.new_traj[-2][0]-x1
            last_y = self.new_traj[-2][1]-y1
        X.append([last_x, last_y, 0, 0, 0])
        sum_ped = 0
        cluster.append(sum_ped)
        sum_ped += 1
        # Insert other person
        for other,ovalue in self.frame_data[self.frame_number].items():
            if other!=self.agent_id:
                x2=ovalue[0]
                y2=ovalue[1]
                dis=math.sqrt((x1-x2)**2+(y1-y2)**2)
                if dis<self.radius:  # 只考虑当前半径范围内的行人
                    # 每个行人最多取past_len个历史点向量
                    frame_id=self.frame_number
                    len_nodes=0
                    while_flag=0   # 一个不会让聚类编号凭空增加的flag
                    while(frame_id>=self.frame_interval and len_nodes<self.graph_obs_past_len and  (other in self.frame_data[frame_id-self.frame_interval])):
                        start_x=self.frame_data[frame_id-self.frame_interval][other][0]-x1
                        start_y=self.frame_data[frame_id-self.frame_interval][other][1]-y1
                        end_x=self.frame_data[frame_id][other][0]-x1
                        end_y=self.frame_data[frame_id][other][1]-y1

                        # 计算当前历史向量终点的相对agent的前后,前为1，后为0
                        front_flag=1
                        if len(self.new_traj) == 1:
                            last_v = goal
                        else:
                            last_v = last_speed
                        ang=self.clockwise_angle(last_v, [end_x,end_y])
                        if ang>=90 and ang<=270:
                            front_flag=0

                        X.append([start_x,start_y,end_x,end_y,front_flag])
                        cluster.append(sum_ped)
                        while_flag=1

                        if len_nodes>0:
                            link_start=len(X)-1
                            edge_index[0].append(link_start)
                            edge_index[1].append(link_start-1)

                        len_nodes += 1
                        frame_id -= self.frame_interval
                    if while_flag == 1:
                        sum_ped += 1
        assert len(cluster)!=0
        X = np.array(X)
        cluster = np.array(cluster)
        X = np.vstack([X, np.zeros((self.padd_to_number - cluster.max()-1, self.graph_feature_len), dtype=X .dtype)])
        cluster = np.hstack([cluster, np.arange(cluster.max()+1, self.padd_to_number)])
        g_data = GraphData(
            x=torch.tensor(X,dtype=torch.float32),
            cluster=torch.tensor(cluster,dtype=torch.int64),
            edge_index=torch.tensor(edge_index,dtype=torch.int64),
            valid_len=torch.tensor([cluster.max()+1]),
            time_step_len=torch.tensor([self.padd_to_number]),
            goal=torch.tensor(goal,dtype=torch.float32),
            last_speed=torch.tensor(last_speed,dtype=torch.float32)
        )
        return g_data

    
    def _update_frame_number(self):
        self.frame_number += self.frame_interval

    # angle by x axis
    def _angle_by_x(self, v):
        x1,y1 = 1,0
        x2,y2 = v
        dot = x1*x2+y1*y2
        det = x1*y2-y1*x2
        theta = np.arctan2(det, dot)
        theta = theta if theta>0 else 2*np.pi+theta
        theta = theta*180/np.pi
        return theta%360
    
    # clockwise angle from v1 to v2
    def _clockwise_angle(self,v1, v2):
        x1,y1 = v1
        x2,y2 = v2
        dot = x1*x2+y1*y2
        det = x1*y2-y1*x2
        theta = np.arctan2(det, dot)
        theta = theta if theta>0 else 2*np.pi+theta
        theta = theta*180/np.pi
        return theta
    
    def _frame_continues_check(self):
        # check if trajectorys is continues
        check = True
        if self.agent_id not in self.trajectorys.keys():
            return False
        start_frame = self.trajectorys[self.agent_id][0][2]
        end_frame = self.trajectorys[self.agent_id][-1][2]
        for i in range(int(start_frame/self.frame_interval),int(end_frame/self.frame_interval)+1):
            if self.agent_id not in self.frame_data[i*self.frame_interval]:
                check = False
                break
        return check

    


        



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='args for SocialGAIL')
    parser.add_argument('--dataset_path',default="./datasets/gc_homo_trajectory.pkl")
    parser.add_argument('--regions',default=16)
    parser.add_argument('--radius',default=6)
    parser.add_argument('--frame_interval',default=20)
    parser.add_argument('--time_interval',default=0.8)
    parser.add_argument('--map_size_bound',default=[-10,40,-20,50]) # [low_x, high_x, low_y, high_y] (int)
    parser.add_argument('--with_last_speed',default=False)
    args = parser.parse_args()


    # # 用于检查自定义的gym环境
    # # 如果你安装了pytorch，则使用上面的，如果你安装了tensorflow，则使用from stable_baselines.common.env_checker import check_env
    # from stable_baselines3.common.env_checker import check_env 
    # env = CrowdEnv(args)
    # check_env(env)




    from stable_baselines3 import PPO
    env = CrowdEnv(args)
    
    model = PPO("MlpPolicy", env, verbose=1,tensorboard_log="./PPO/")
    model.learn(total_timesteps=5000000)   # 400000

    # obs = env.reset()
    # # 验证十次
    # for i in range(400):
    #     action, state = model.predict(observation=obs)
    #     obs, reward, done, info = env.step(action)
    #     env.render(i)
    #     if done:
    #         obs = env.reset()
    # env.draw_gif()