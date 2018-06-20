# Christopher Iliffe Sprague
# christopher.iliffe.sprague@gmail.com

import numpy as np, seaborn as sb

class Node(object):

    def __init__(self, tasks):

        # assign conditions and actions
        self.tasks = tasks
        self.reset_response()

    def reset_response(self):

        # extract actions and conditions
        self.response = dict()
        for task in self.tasks:
            if isinstance(task, Node):
                self.response.update(task.response)
            else:
                self.response[task.__name__] = 3

class Sequence(Node):

    def __init__(self, tasks):

        # initialise base
        Node.__init__(self, tasks)

    def __call__(self):

        # reset response
        self.reset_response()

        # examine conditions
        for task in self.tasks:

            # compute status
            status = task()
            if isinstance(task, Node):
                self.response.update(task.response)
            else:
                self.response[task.__name__] = status

            # if true
            if status is 1:
                continue

            # if false
            elif status is 0:
                return 0

            # if running
            elif status == 2:
                return 2

        # all conditions are true
        return 1

class Fallback(Node):

    def __init__(self, tasks):

        # initialise base
        Node.__init__(self, tasks)

    def __call__(self):

        # response list
        self.reset_response()

        # examine conditions
        for task in self.tasks:

            # compute status
            status = task()
            if isinstance(task, Node):
                self.response.update(task.response)
            else:
                self.response[task.__name__] = status

            # if true
            if status is 1:
                return 1

            # if false
            elif status is 0:
                continue

            # if running
            elif status is 2:
                return 2

        # all conditions are false
        return 0
