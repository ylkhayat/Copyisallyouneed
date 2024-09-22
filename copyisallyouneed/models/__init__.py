from .agent import Agent
from .copyisallyouneed import Copyisallyouneed

def load_model(args):
    model_name = args['models'][args['model']]['model_name']
    model = globals()[model_name](**args)
    agent = Agent(model, args)
    return agent
