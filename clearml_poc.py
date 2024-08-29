from os import environ as env  # Import environ from the os module to access environment variables.
import numpy as np

# Check if ClearML and remote execution are allowed based on environment variables.
ALLOW_CLEARML = True if env.get("ALLOW_CLEARML") == "yes" else False
RUNNING_REMOTE = True if env.get("RUNNING_REMOTE") == "yes" else False

# Initialize ClearML for experiment tracking
def clearml_init(args=None, params=None):
    global execution_task  # Declare execution_task as a global variable to be accessed within other functions.
    global output_model  # Declare output_model as a global variable for model tracking.

    if ALLOW_CLEARML:
        from clearml import Task, OutputModel  # Import ClearML's Task and OutputModel classes.

        # Initialize a ClearML task for tracking experiments
        execution_task = Task.init(
            project_name="DeepBiologyRnaCompete",  # Set the project name in ClearML
            task_name="hidden layers - match an entity to another sentence to detect same entity",  # Task name
            task_type=Task.TaskTypes.testing,  # Task type (e.g., training, testing)
        )

        # Create an OutputModel instance for handling model saving and versioning
        output_model = OutputModel(task=execution_task, framework='PyTorch')

        # If running locally, prompt for a task description and set parameters
        if execution_task.running_locally():
            name = input("enter description for task:\n")
            execution_task.set_name(name)  # Set the name of the task based on user input
            if params:
                # Combine parameters from the args and params objects and set them in the task
                all_params = dict(dict(params.__dict__) if params else {}) | vars(args or {})
                execution_task.set_parameters_as_dict(all_params)

        # If running remotely, execute the task on a GPU queue
        if RUNNING_REMOTE:
            execution_task.execute_remotely(queue_name="gpu", exit_process=True)

# Function to log images to ClearML
def clearml_display_image(image, iteration, series, description):
    if ALLOW_CLEARML:
        # Log an image with a description, iteration, and series name
        execution_task.get_logger().report_image(
            description,
            image=image,
            iteration=iteration,
            series=series
        )

# Function to log scalar values (e.g., metrics) to ClearML
def add_point_to_graph(title, series, x, y):
    if ALLOW_CLEARML:
        # Log a scalar value for a given title and series
        execution_task.get_logger().report_scalar(title, series, value=y, iteration=x)

# Function to log scatter plots to ClearML
def add_scatter(title, series, iteration, values):
    if ALLOW_CLEARML:
        numpy_values = np.array(values)  # Convert the values to a NumPy array
        if numpy_values.ndim == 1:
            # If the array is 1D, add an x-dimension based on the index of each value
            values = np.column_stack((np.arange(len(numpy_values)), numpy_values))
        # Log the scatter plot with lines and markers
        execution_task.get_logger().report_scatter2d(
            title, series, scatter=values, iteration=iteration,
            mode='lines+markers'
        )

# Function to log Matplotlib figures to ClearML
def add_matplotlib(figure, iteration, series):
    if ALLOW_CLEARML:
        # Log a Matplotlib figure for a given iteration and series
        execution_task.get_logger().report_matplotlib(
            figure, iteration=iteration, series=series
        )

# Function to log confusion matrices to ClearML
def add_confusion_matrix(matrix, title, series, iteration):
    if ALLOW_CLEARML:
        # Log a confusion matrix with a title and series for a specific iteration
        execution_task.get_logger().report_confusion_matrix(
            title, series=series, matrix=matrix, iteration=iteration
        )

# Function to log text outputs to ClearML
def add_text(text):
    if ALLOW_CLEARML:
        # Log a text output to the ClearML task
        execution_task.get_logger().report_text(text)

# Function to retrieve parameters from ClearML
def get_param(param_name, section='General'):
    if ALLOW_CLEARML:
        # Retrieve a parameter value from the ClearML task configuration
        return execution_task.get_parameter(f'{section}/{param_name}')
    return None

# Function to upload a model to ClearML
def upload_model_to_clearml(model_path, params):
    if ALLOW_CLEARML:
        # Update the model path and save the model with parameters to ClearML
        execution_task.update_output_model(model_path=model_path)
        output_model.update_design(config_dict=dict(vars(params)))
