# -*- coding: utf-8 -*-

def sim(env,agent,max_steps=1000):
    
    # reset environment
    state = env.reset()
    
    for steps in range(max_steps):
        
        env.render()
        
        # sample action from the current policy
        action = agent.get_action(state)
        
        # step the environment
        next_state, reward, done, info = env.step(action)
        
        # step the agent
        agent.step(state, action, reward, next_state, done)
        
        # update state
        state = next_state
        
        # terminate if done
        if done:
            break
            
    env.close()
