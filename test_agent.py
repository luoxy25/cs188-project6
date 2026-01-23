import model
import deepQLearningAgents
import layout

l = layout.getLayout("smallGrid")
agent = deepQLearningAgents.PacmanDeepQAgent(layout_input=l)
print(f"numTrainingGames: {agent.model.numTrainingGames}")
