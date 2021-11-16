"""
CS6476 Problem Set 5 imports. Only Numpy and cv2 are allowed.
"""
import numpy as np
import cv2


# Assignment code
class KalmanFilter(object):
    """A Kalman filter tracker"""

    def __init__(self, init_x, init_y, Q=0.1 * np.eye(4), R=0.1 * np.eye(2)):
        """Initializes the Kalman Filter

        Args:
            init_x (int or float): Initial x position.
            init_y (int or float): Initial y position.
            Q (numpy.array): Process noise array.
            R (numpy.array): Measurement noise array.
        """
        self.state = np.array([init_x, init_y, 0., 0.])  # state
        self.process_noise = Q.astype(float)
        self.measurement_noise = R.astype(float)
        self.covariance_matrix = np.eye(4).astype(float)
        self.transition_matrix = np.eye(4).astype(float)
        self.measurement_matrix = np.eye(2, 4)
        self.kalman_gain = None

    def predict(self):
        self.state = np.dot(self.transition_matrix, self.state)
        self.covariance_matrix = self.transition_matrix * self.covariance_matrix * \
                                 np.transpose(self.transition_matrix) + self.process_noise

    def correct(self, meas_x, meas_y):
        self.kalman_gain = np.dot(np.dot(self.covariance_matrix, np.transpose(self.measurement_matrix)), np.linalg.inv(np.dot(self.measurement_matrix, np.dot(self.covariance_matrix, np.transpose(self.measurement_matrix))) + self.measurement_noise))
        self.state = self.state + np.dot(self.kalman_gain, np.array([meas_x, meas_y]) - np.dot(self.measurement_matrix, self.state))
        prod = np.dot(self.kalman_gain, self.measurement_matrix)
        self.covariance_matrix = np.dot(np.eye(prod.shape[0], prod.shape[1]) - prod, self.covariance_matrix)

    def process(self, measurement_x, measurement_y):

        self.predict()
        self.correct(measurement_x, measurement_y)

        return self.state[0], self.state[1]


class ParticleFilter(object):
    """A particle filter tracker.

    Encapsulating state, initialization and update methods. Refer to
    the method run_particle_filter( ) in experiment.py to understand
    how this class and methods work.
    """

    def __init__(self, frame, template, **kwargs):
        """Initializes the particle filter object.

        The main components of your particle filter should at least be:
        - self.particles (numpy.array): Here you will store your particles.
                                        This should be a N x 2 array where
                                        N = self.num_particles. This component
                                        is used by the autograder so make sure
                                        you define it appropriately.
                                        Make sure you use (x, y)
        - self.weights (numpy.array): Array of N weights, one for each
                                      particle.
                                      Hint: initialize them with a uniform
                                      normalized distribution (equal weight for
                                      each one). Required by the autograder.
        - self.template (numpy.array): Cropped section of the first video
                                       frame that will be used as the template
                                       to track.
        - self.frame (numpy.array): Current image frame.

        Args:
            frame (numpy.array): color BGR uint8 image of initial video frame,
                                 values in [0, 255].
            template (numpy.array): color BGR uint8 image of patch to track,
                                    values in [0, 255].
            kwargs: keyword arguments needed by particle filter model:
                    - num_particles (int): number of particles.
                    - sigma_exp (float): sigma value used in the similarity
                                         measure.
                    - sigma_dyn (float): sigma value that can be used when
                                         adding gaussian noise to u and v.
                    - template_rect (dict): Template coordinates with x, y,
                                            width, and height values.
        """
        self.num_particles = kwargs.get('num_particles')  # required by the autograder
        self.sigma_exp = kwargs.get('sigma_exp')  # required by the autograder
        self.sigma_dyn = kwargs.get('sigma_dyn')  # required by the autograder
        self.template_rect = kwargs.get('template_coords')  # required by the autograder
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #
        # The way to do it is:
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)

        self.template = template
        self.gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        self.frame = frame

        self.particles = None # Initialize your particles array. Read the docstring.
        self.weights = 1/self.num_particles * np.ones(self.num_particles)
        # Get the center of the template as the state
        self.state = [self.template_rect["x"] + self.template_rect["w"]//2, self.template_rect["y"] + self.template_rect["h"]//2]
        # Initialize your weights array. Read the docstring.
        # Initialize any other components you may need when designing your filter.

    def get_particles(self):
        """Returns the current particles state.

        This method is used by the autograder. Do not modify this function.

        Returns:
            numpy.array: particles data structure.
        """
        return self.particles

    def get_weights(self):
        """Returns the current particle filter's weights.

        This method is used by the autograder. Do not modify this function.

        Returns:
            numpy.array: weights data structure.
        """
        return self.weights

    def get_error_metric(self, template, frame_cutout):
        """Returns the error metric used based on the similarity measure.

        Returns:
            float: similarity value.
        """
        norm = template.shape[0] * template.shape[1]
        return np.sum((template.astype(float) - frame_cutout.astype(float))**2) / norm

    def resample_particles(self):
        """Returns a new set of particles

        This method does not alter self.particles.

        Use self.num_particles and self.weights to return an array of
        resampled particles based on their weights.

        See np.random.choice or np.random.multinomial.
        
        Returns:
            numpy.array: particles data structure.
        """
        return np.random.choice(a = self.num_particles, size = self.num_particles, p=self.weights)

    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        Implement the particle filter in this method returning None
        (do not include a return call). This function should update the
        particles and weights data structures.

        Make sure your particle filter is able to cover the entire area of the
        image. This means you should address particles that are close to the
        image borders.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame,
                                 values in [0, 255].

        Returns:
            None.
        """
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rand_x = np.random.normal(self.state[0], self.sigma_dyn, size = self.num_particles)
        rand_y = np.random.normal(self.state[1], self.sigma_dyn, size = self.num_particles)
        self.particles = np.stack((rand_x, rand_y), axis = -1)
        for particle in self.particles:
            particle[0] = np.clip(particle[0], 0, frame.shape[1])
            particle[1] = np.clip(particle[1], 0, frame.shape[0])

        self.weights = []
        for particle in self.particles:
            frame_temp = self.get_patch(particle, gray_frame)
            mse = self.get_error_metric(self.gray_template, frame_temp)
            sim_value = np.exp(-mse / (2.0 * self.sigma_exp ** 2))
            self.weights.append(sim_value)

        self.weights /= np.sum(self.weights)
        idx = np.argmax(self.weights)
        self.state = self.particles[idx]
        self.particles = self.particles[self.resample_particles()]

    def render(self, frame_in):
        """Visualizes current particle filter state.

        This method may not be called for all frames, so don't do any model
        updates here!

        These steps will calculate the weighted mean. The resulting values
        should represent the tracking window center point.

        In order to visualize the tracker's behavior you will need to overlay
        each successive frame with the following elements:

        - Every particle's (x, y) location in the distribution should be
          plotted by drawing a colored dot point on the image. Remember that
          this should be the center of the window, not the corner.
        - Draw the rectangle of the tracking window associated with the
          Bayesian estimate for the current location which is simply the
          weighted mean of the (x, y) of the particles.
        - Finally we need to get some sense of the standard deviation or
          spread of the distribution. First, find the distance of every
          particle to the weighted mean. Next, take the weighted sum of these
          distances and plot a circle centered at the weighted mean with this
          radius.

        This function should work for all particle filters in this problem set.

        Args:
            frame_in (numpy.array): copy of frame to render the state of the
                                    particle filter.
        """
        if self.particles is None:
            return
        x_weighted_mean = 0
        y_weighted_mean = 0

        for i in range(self.num_particles):
            x_weighted_mean += self.particles[i, 0] * self.weights[i]
            y_weighted_mean += self.particles[i, 1] * self.weights[i]

        # Complete the rest of the code as instructed.

        cv2.rectangle(frame_in, (int(self.state[0] - self.template.shape[1] / 2), int(self.state[1] - self.template.shape[0] / 2)), (int(self.state[0] + self.template.shape[1] / 2), int(self.state[1] + self.template.shape[0] / 2)), (255, 255, 255), 2)
        spread = 0
        for particle, weight in zip(self.particles, self.weights):
            cv2.circle(frame_in, tuple(particle.astype(int)), 2, (255, 0, 0), 1)
            spread += weight * np.sqrt((x_weighted_mean - particle[0])**2 + (y_weighted_mean - particle[1])**2)

        cv2.circle(frame_in, tuple(self.state.astype(int)), int(spread), (255, 255, 255), 1)

    def get_patch(self, particle, frame, shape = None):
        if shape is None:
            shape = self.template.shape[0], self.template.shape[1]
        x1 = int(particle[0] - shape[1]/2)
        y1 = int(particle[1] - shape[0]/2)
        x2 = x1 + shape[1]
        y2 = y1 + shape[0]

        if x1 < 0:
            x1 = 0
            x2 = shape[1]
        if x2 > frame.shape[1]:
            x2 = frame.shape[1]
            x1 = x2 - shape[1]
        if y1 < 0:
            y1 = 0
            y2 = shape[0]
        if y2 > frame.shape[0]:
            y2 = frame.shape[0]
            y1 = y2 - shape[0]

        return frame[y1:y2, x1:x2]


class AppearanceModelPF(ParticleFilter):
    """A variation of particle filter tracker."""

    def __init__(self, frame, template, **kwargs):
        """Initializes the appearance model particle filter.

        The documentation for this class is the same as the ParticleFilter
        above. There is one element that is added called alpha which is
        explained in the problem set documentation. By calling super(...) all
        the elements used in ParticleFilter will be inherited so you do not
        have to declare them again.
        """

        super(AppearanceModelPF, self).__init__(frame, template, **kwargs)  # call base class constructor

        self.alpha = kwargs.get('alpha')  # required by the autograder
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #
        # The way to do it is:
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)

    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        This process is also inherited from ParticleFilter. Depending on your
        implementation, you may comment out this function and use helper
        methods that implement the "Appearance Model" procedure.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame, values in [0, 255].

        Returns:
            None.
        """
        super(AppearanceModelPF, self).process(frame)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_temp = self.get_patch(self.state, gray_frame)
        self.gray_template = self.alpha*frame_temp.astype(np.float32) + (1.-self.alpha)*self.gray_template.astype(np.float32)


class MDParticleFilter(AppearanceModelPF):
    """A variation of particle filter tracker that incorporates more dynamics."""

    def __init__(self, frame, template, **kwargs):
        """Initializes MD particle filter object.

        The documentation for this class is the same as the ParticleFilter
        above. By calling super(...) all the elements used in ParticleFilter
        will be inherited so you don't have to declare them again.
        """

        super(MDParticleFilter, self).__init__(frame, template, **kwargs)  # call base class constructor
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #
        # The way to do it is:
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)
        self.resized_template = self.gray_template
        self.scale = kwargs.get("scale") or 0.9999
        self.update_scale = kwargs.get("scale") or 0.996
        self.mse_threshold = kwargs.get("mse") or 7300

    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        This process is also inherited from ParticleFilter. Depending on your
        implementation, you may comment out this function and use helper
        methods that implement the "More Dynamics" procedure.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame,
                                 values in [0, 255].

        Returns:
            None.
        """
        self.resized_template = cv2.resize(self.gray_template, dsize = (0, 0), fx = self.scale, fy = self.scale)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rand_x = np.random.normal(self.state[0], self.sigma_dyn, size = self.num_particles)
        rand_y = np.random.normal(self.state[1], self.sigma_dyn, size = self.num_particles)
        self.particles = np.stack((rand_x, rand_y), axis = -1)
        for particle in self.particles:
            particle[0] = np.clip(particle[0], 0, frame.shape[1])
            particle[1] = np.clip(particle[1], 0, frame.shape[0])

        self.weights = []
        mses = []
        for particle in self.particles:
            frame_temp = self.get_patch(particle, gray_frame, self.resized_template.shape)
            mse = self.get_error_metric(self.resized_template, frame_temp)
            mses.append(mse)
            sim_value = np.exp(-mse / (2.0 * self.sigma_exp ** 2))
            self.weights.append(sim_value)

        avg = np.average(mses)
        self.weights /= np.sum(self.weights)
        if avg < self.mse_threshold:
            # idx = np.argmax(self.weights)
            # self.state = self.particles[idx]
            self.state = np.average(self.particles, axis = 0, weights = self.weights)
        else:
            print("Occlusion")
        self.particles = self.particles[self.resample_particles()]
        self.scale *= self.update_scale

    def render(self, frame_in):
        """Visualizes current particle filter state.

        This method may not be called for all frames, so don't do any model
        updates here!

        These steps will calculate the weighted mean. The resulting values
        should represent the tracking window center point.

        In order to visualize the tracker's behavior you will need to overlay
        each successive frame with the following elements:

        - Every particle's (x, y) location in the distribution should be
          plotted by drawing a colored dot point on the image. Remember that
          this should be the center of the window, not the corner.
        - Draw the rectangle of the tracking window associated with the
          Bayesian estimate for the current location which is simply the
          weighted mean of the (x, y) of the particles.
        - Finally we need to get some sense of the standard deviation or
          spread of the distribution. First, find the distance of every
          particle to the weighted mean. Next, take the weighted sum of these
          distances and plot a circle centered at the weighted mean with this
          radius.

        This function should work for all particle filters in this problem set.

        Args:
            frame_in (numpy.array): copy of frame to render the state of the
                                    particle filter.
        """
        x_weighted_mean = 0
        y_weighted_mean = 0

        for i in range(self.num_particles):
            x_weighted_mean += self.particles[i, 0] * self.weights[i]
            y_weighted_mean += self.particles[i, 1] * self.weights[i]

        # Complete the rest of the code as instructed.

        cv2.rectangle(frame_in, (int(self.state[0] - self.resized_template.shape[1] / 2), int(self.state[1] - self.resized_template.shape[0] / 2)), (int(self.state[0] + self.resized_template.shape[1] / 2), int(self.state[1] + self.resized_template.shape[0] / 2)), (255, 255, 255), 2)
        spread = 0
        for particle, weight in zip(self.particles, self.weights):
            cv2.circle(frame_in, tuple(particle.astype(int)), 2, (255, 0, 0), 1)
            spread += weight * np.sqrt((x_weighted_mean - particle[0])**2 + (y_weighted_mean - particle[1])**2)

        cv2.circle(frame_in, tuple(self.state.astype(int)), int(spread), (255, 255, 255), 1)
