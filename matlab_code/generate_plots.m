% algorithm: reinforcement learning 
% task: trajectoryTracking

%% tracking performance
clc, clear all, close all,

dir_path = '/home/jhon/reinforcement_learning/ppo_dm_control/training';
task_name='standup';
data_path = fullfile(dir_path, task_name, "states");

% get performance data
data = readtable(data_path, 'PreserveVariableNames', true);

% time range
t_start=1;
t_step=1;
t_end=size(data, 1);

t=0:0.01:5;
% reference
%pos_ref = pi/2 + pi/4*sin(2*pi*t);
%vel_ref  = pi/4*2*pi*cos(2*pi*t);
%accel_ref = -pi/4*2*pi*2*pi*sin(2*pi*t);
pos_ref = data.pos_des(t_start : t_step : t_end);
vel_ref = data.vel_des(t_start : t_step : t_end);
accel_ref = data.accel_des(t_start : t_step : t_end);

% measured
pos_med = data.pos_med(t_start : t_step : t_end);
vel_med = data.vel_med(t_start : t_step : t_end);
accel_med = data.accel_med(t_start : t_step : t_end);

% position
figure(1), grid on, box on, hold on
    plot(t, pos_ref, 'r', LineWidth=2, LineStyle='-'), hold on
    plot(t, pos_med, 'k', LineWidth=2, LineStyle='--')

% velocity
figure(2), grid on, box on, hold on
    plot(t, vel_ref, 'r', LineWidth=2, LineStyle='-'), hold on
    plot(t, vel_med, 'k', LineWidth=2, LineStyle='--')

% acceleration
figure(3), grid on, box on, hold on
    plot(t, accel_ref, 'r', LineWidth=2, LineStyle='-'), hold on
    plot(t, accel_med, 'k', LineWidth=2, LineStyle='--')    


%% reward 
clc, clear all, close all,

dir_path = '/home/jhon/reinforcement_learning/ppo_dm_control/training/hopper';
task_name='walkEnv';
data_path = fullfile(dir_path, task_name, "data");

% get performance data
data = readtable(data_path, 'PreserveVariableNames', true);
% time range
t_start=1;
t_step=1;
t_end=size(data, 1);
% steps
steps=t_start:t_step:t_end;

% mean and std
mean = data.mean(t_start:t_step:t_end)';
max_mean = data.mean(t_start:t_step:t_end)' + 1*data.std(t_start:t_step:t_end)';
min_mean  = data.mean(t_start:t_step:t_end)' - 1*data.std(t_start:t_step:t_end)';

% useful vectors
v_max = [steps', max_mean'];

v_min  = [fliplr(steps)', fliplr(min_mean)'];
v = [v_max; v_min];
f = 1:size(v,1);

figure(2), grid on, box on, hold on
    patch('faces', f, 'Vertices', v, 'FaceColor', 'red', 'FaceAlpha', 0.3), hold on
    plot(steps, mean, 'k', 'LineWidth',3, 'LineStyle','-'), hold on
    plot(steps, max_mean, 'r','LineWidth',2), hold on
    plot(steps, min_mean, 'r', 'LineWidth',2), hold on


    
    






