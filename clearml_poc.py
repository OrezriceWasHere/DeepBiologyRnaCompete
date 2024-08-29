import json
from os import environ as env
import numpy as np

ALLOW_CLEARML = True if env.get("ALLOW_CLEARML") == "yes" else False
RUNNING_REMOTE = True if env.get("RUNNING_REMOTE") == "yes" else False

def clearml_init(args=None, params=None):
    global execution_task
    global output_model
    if ALLOW_CLEARML:

        from clearml import Task, OutputModel
        # Task.add_requirements("requirements.txt")
        execution_task = Task.init(project_name="DeepBiologyRnaCompete",

                                   task_name="hidden layers - match an entity to another sentence to detect same entity",
                                   task_type=Task.TaskTypes.testing,
                                   )
        output_model = OutputModel(task=execution_task, framework='PyTorch')


        if execution_task.running_locally():
            name = input("enter description for task:\n")
            execution_task.set_name(name)
            if params:
                all_params = dict(dict(params.__dict__) if params else {}) | vars(args or {})
                execution_task.set_parameters_as_dict(all_params)

        if RUNNING_REMOTE:
            execution_task.execute_remotely(queue_name="gpu", exit_process=True)


def clearml_display_image(image, iteration, series, description):
    if ALLOW_CLEARML:
        execution_task.get_logger().report_image(description,
                                                 image=image,
                                                 iteration=iteration,
                                                 series=series)


def add_point_to_graph(title, series, x, y):
    if ALLOW_CLEARML:
        execution_task.get_logger().report_scalar(title, series, value=y, iteration=x)


def add_scatter(title, series, iteration, values):
    if ALLOW_CLEARML:
        numpy_values = np.array(values)
        if numpy_values.ndim == 1:
            # adding x dimension to be indexed of values (first value - 0, second value - 1, etc.)
            values = np.column_stack((np.arange(len(numpy_values)), numpy_values))
        execution_task.get_logger().report_scatter2d(title, series, scatter=values, iteration=iteration,
                                                     mode='lines+markers')


def add_matplotlib(figure, iteration, series):
    if ALLOW_CLEARML:
        execution_task.get_logger().report_matplotlib(figure, iteration=iteration, series=series)


def add_confusion_matrix(matrix, title, series, iteration):
    if ALLOW_CLEARML:
        execution_task.get_logger().report_confusion_matrix(title, series=series, matrix=matrix, iteration=iteration)


def add_text(text):
    if ALLOW_CLEARML:
        execution_task.get_logger().report_text(text)


def get_param(param_name, section='General'):
    if ALLOW_CLEARML:
        return execution_task.get_parameter(f'{section}/{param_name}')
    return None



def upload_model_to_clearml(model_path,  params):
    if ALLOW_CLEARML:

        execution_task.update_output_model(model_path=model_path)
        output_model.update_design(config_dict=dict(vars(params)))