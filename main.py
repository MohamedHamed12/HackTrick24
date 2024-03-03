from repo.Solvers.fox_submission_solver import init_fox
from repo.Solvers.riddle_solver import problem_solving_hard
# Example usage
# team_id='ds42W0d'
team_id = "your_team_id"
# message = init_fox(team_id)
# print("Message:", message)

m,n = map(int, input("m n: ").split())
print(problem_solving_hard(m,n))