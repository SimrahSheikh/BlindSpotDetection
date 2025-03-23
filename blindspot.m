clc; clear; close all;

%% **Step 1: Simulate LiDAR Point Cloud with Fog**
numPoints = 500;
x = rand(numPoints,1) * 20 - 10; 
y = rand(numPoints,1) * 20 - 10;
z = rand(numPoints,1) * 2;

% Add fog noise (reduces accuracy)
fogIntensity = 0.3;
noise = fogIntensity * randn(numPoints, 3); 
pointCloudData = [x y z] + noise;

% Reduce intensity for far objects (simulate LiDAR attenuation)
intensity = exp(-0.05 * sqrt(x.^2 + y.^2)); 

% Plot foggy LiDAR point cloud
figure;
scatter3(pointCloudData(:,1), pointCloudData(:,2), pointCloudData(:,3), 10, intensity, 'filled');
xlabel('X-axis (m)'); ylabel('Y-axis (m)'); zlabel('Z-axis (m)');
title('LiDAR Point Cloud with Fog');
colorbar;
grid on;

%% **Step 2: Simulate Radar Detections with Rain Noise**
numObjects = 5;
objectDistance = randi([5, 50], numObjects, 1);
objectAngle = randi([-45, 45], numObjects, 1);
objectVelocity = randi([-10, 20], numObjects, 1); % Object speed in m/s

% Introduce random noise due to rain
rainNoise = 0.2 * randi([-5, 5], numObjects, 1); 
objectDistance = objectDistance + rainNoise;

% Reduce detection probability in heavy rain
rainIntensity = 0.5;
detected(1) = 1

% Apply rain effect by keeping only detected objects
objectDistance = objectDistance .* detected;
objectAngle = objectAngle .* detected;
objectVelocity = objectVelocity .* detected;

% Plot radar detections under rainy conditions
figure;
polarplot(deg2rad(objectAngle), objectDistance, 'bo');
title('Radar Detections in Rain');

%% **Step 3: Generate Foggy and Rainy Camera Image**
image = ones(416, 416, 3) * 0.6;

% Apply fog overlay effect
fogOverlay = ones(size(image)) * 0.8;
alphaFog = 0.5;
foggyImage = (1 - alphaFog) * image + alphaFog * fogOverlay;

% Generate rain streaks
numRaindrops = 700;
for i = 1:numRaindrops
    x = randi([1, 416]); 
    y = randi([1, 416]);
    len = randi([10, 30]); 
    foggyImage = insertShape(foggyImage, 'Line', [x y x y+len], 'Color', 'white', 'LineWidth', 1);
end

% Display foggy and rainy camera image
figure;
imshow(foggyImage);
title('Camera View in Fog and Rain');

%% **Step 4: Simulate Sensor Measurements and Apply Kalman Filter**
truePosition = [10, 5]; % Ground truth position in meters
vehicleVelocity = 15; % Vehicle speed (m/s)

% Simulate LiDAR measurement with fog-induced noise
lidarNoise = randn(1,2) * 0.5;
lidarMeasurement = truePosition + lidarNoise;

% Convert radar measurements from polar to Cartesian coordinates
validIndices = find(detected == 1); % Get indices of detected objects
if ~isempty(validIndices)
    range = objectDistance(validIndices(1));
    angle = objectAngle(validIndices(1));
    velocity = objectVelocity(validIndices(1));
else
    range = NaN; % No object detected
    angle = NaN;
    velocity = NaN;
end

velocity = objectVelocity(1); % Object velocity

radarX = range * cosd(angle);
radarY = range * sind(angle);
radarMeasurement = [radarX, radarY];

%% **Step 5: Initialize and Run Kalman Filter for Sensor Fusion**
% Initialize Kalman Filter
kf = trackingKF('MotionModel', '2D Constant Velocity');

% Use a 4x4 state covariance matrix (position + velocity tracking)
kf.StateCovariance = eye(4);

% Adjust measurement noise covariance matrix (2x2)
kf.MeasurementNoise = diag([0.8, 1.5]); 

% Adjust process noise for smoother state estimation (2x2)
kf.ProcessNoise = diag([0.1, 0.1]);

% Predict the next state
predictedState = predict(kf);

% Update the Kalman Filter using sensor fusion
if ~isnan(radarMeasurement(1)) && ~isnan(radarMeasurement(2))
    fusionData = mean([lidarMeasurement; radarMeasurement], 1);
else
    fusionData = lidarMeasurement; % If radar fails, rely only on LiDAR
end

updatedState = correct(kf, fusionData);

%% **Step 6: Display Results**
fprintf('\nSensor Fusion Results:\n');
fprintf('True Position: [%.2f, %.2f]\n', truePosition);
fprintf('LiDAR Measurement: [%.2f, %.2f]\n', lidarMeasurement);
fprintf('Radar Measurement: [%.2f, %.2f]\n', radarMeasurement);
fprintf('Fused State Estimate (Kalman Output): [%.2f, %.2f]\n', updatedState(1:2));
fprintf('Vehicle Velocity: %.2f m/s\n', vehicleVelocity);
fprintf('Detected Object Velocity: %.2f m/s\n', velocity);
fprintf('Detected Object Angle: %.2f degrees\n', angle);

%% **Step 7: Plot Sensor Fusion Results**
figure;
scatter(truePosition(1), truePosition(2), 100, 'g', 'filled'); hold on; 
scatter(lidarMeasurement(1), lidarMeasurement(2), 100, 'r', 'filled'); 
scatter(radarMeasurement(1), radarMeasurement(2), 100, 'b', 'filled'); 
scatter(updatedState(1), updatedState(2), 100, 'k', 'filled'); 

legend('True Position', 'LiDAR Measurement', 'Radar Measurement', 'Fused Position');
xlabel('X-axis (m)'); ylabel('Y-axis (m)');
title('Sensor Fusion: LiDAR, Radar, and Kalman Filter Output');
grid on;
hold off;
