from utils.conversion_utils import BPMN
from utils.grpo_utils import *

    
#Validation reward function
def validation_reward_func(prompts, completions, **kwargs) -> list[float]:
    extracted_responses = [completion[0]['content'] for completion in completions]
    rewards = []
    for i, response in enumerate(extracted_responses):
        try: 
            BPMN(**response)
            rewards.append(2.0)
        except:
            rewards.append(-5.0)
    return rewards

#Task reward function
def task_reward_funct(prompts, completions, **kwargs) -> list[float]:
    extracted_responses = [completion[0]['content'] for completion in completions]
    rewards = []
    for i, response in enumerate(extracted_responses):
        try: 
            BPMN(**response)
            valid = True
        except:
            valid = False
            rewards.append(0)
        if valid:
            _, task_names, _, _ = extract_bpmn_names(response)
            f1 = component_similarity_metric(true_task[i], task_names)
            print(f1)
            if f1 > 0:
                rewards.append(f1*2)
            else:
                rewards.append(-1)
    return rewards

#Actor reward function
def actor_reward_funct(prompts, completions, **kwargs) -> list[float]:
    extracted_responses = [completion[0]['content'] for completion in completions]
    rewards = []
    for i, response in enumerate(extracted_responses):
        try: 
            #BPMN(**response)
            lane_names, _, _, _ = extract_bpmn_names(response)
            f1 = component_similarity_metric(true_actors[i], lane_names)
            print(f1)
            if f1 > 0:
                rewards.append(f1*2)
            else:
                rewards.append(-2)
        except:
            rewards.append(0)
    return rewards

def exist_gateway_reward_func(prompts, completions, **kwargs) -> list[float]:
    extracted_responses = [completion[0]['content'] for completion in completions]
    rewards = []
    for i, response in enumerate(extracted_responses):
            try: 
                BPMN(**response)
                if check_are_there_gateway(response):
                    rewards.append(1.0)
                else:
                    rewards.append(-2.0)
            except:
                rewards.append(0)
    return rewards

def dead_activities_reward_func(prompts, completions, **kwargs) -> list[float]:
    extracted_responses = [completion[0]['content'] for completion in completions]
    rewards = []
    for i, response in enumerate(extracted_responses):
            try: 
                BPMN(**response)
                if check_no_dead_activities(response):
                    rewards.append(1.0)
                else:
                    rewards.append(-1)
            except:
                rewards.append(0)
    return rewards

def exist_end_event_reward_func(prompts, completions, **kwargs) -> list[float]:
    extracted_responses = [completion[0]['content'] for completion in completions]
    rewards = []
    for i, response in enumerate(extracted_responses):
            try: 
                BPMN(**response)
                if check_exist_end_event(response):
                    rewards.append(1.0)
                else:
                    rewards.append(-1)
            except:
                rewards.append(0)
    return rewards

def option_to_complete_reward_func(prompts, completions, **kwargs) -> list[float]:
    extracted_responses = [completion[0]['content'] for completion in completions]
    rewards = []
    for i, response in enumerate(extracted_responses):
            try: 
                BPMN(**response)
                if check_option_to_complete(response):
                    rewards.append(1.0)
                else:
                    rewards.append(-1)
            except:
                rewards.append(0)
    return rewards

def no_infinite_loop_reward_func(prompts, completions, **kwargs) -> list[float]:
    extracted_responses = [completion[0]['content'] for completion in completions]
    rewards = []
    for i, response in enumerate(extracted_responses):
            try: 
                BPMN(**response)
                if check_no_infinite_loop(response):
                    rewards.append(1.0)
                else:
                    rewards.append(-1)
            except:
                rewards.append(0)
    return rewards

def safeness_reward_func(prompts, completions, **kwargs) -> list[float]:
    extracted_responses = [completion[0]['content'] for completion in completions]
    rewards = []
    for i, response in enumerate(extracted_responses):
            try: 
                BPMN(**response)
                if check_safeness(response):
                    rewards.append(1.0)
                else:
                    rewards.append(-1)
            except:
                rewards.append(0)
    return rewards

def correct_gateway_reward_func(prompts, completions, **kwargs) -> list[float]:
    extracted_responses = [completion[0]['content'] for completion in completions]
    rewards = []
    for i, response in enumerate(extracted_responses):
            try: 
                BPMN(**response)
                if check_gateway_correctness(response):
                    rewards.append(1.0)
                else:
                    rewards.append(-1)
            except:
                rewards.append(0)
    return rewards

def sincronization_reward_funct(prompts, completions, **kwargs) -> list[float]:
    extracted_responses = [completion[0]['content'] for completion in completions]
    rewards = []
    for i, response in enumerate(extracted_responses):
            try: 
                BPMN(**response)
                if check_proper_synchronization(response):
                    rewards.append(1.0)
                else:
                    rewards.append(-1)
            except:
                rewards.append(0)
    return rewards

def task_type_reward_funct(prompts, completions, **kwargs) -> list[float]:
    extracted_responses = [completion[0]['content'] for completion in completions]
    rewards = []
    for i, response in enumerate(extracted_responses):
        try: 
            BPMN(**response)
            _, _, task_types, _ = extract_bpmn_names(response)
            prop = (len(task_types) - task_types.count('task'))/len(task_types)
            if prop > 0:
                rewards.append(prop*2)
            else:
                rewards.append(-2)
        except:
            rewards.append(0)
    return rewards
