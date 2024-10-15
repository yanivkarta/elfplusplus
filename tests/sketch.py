from typing import List
# Write any import statements here

def getMaxDamageDealt(N: int, H: List[int], D: List[int], B: int) -> float:
  # Write your code here
  max_damage = 0
  max_impact = 0
  impact_selection ={}
  damage_selection ={}
  W = 0
  Z = 0
  X = 1
  Y = 1
  for impact, damage in zip(H,D):

    max_impact = max(impact, max_impact)
    max_damage = max(damage, max_damage)
    
    if impact not in impact_selection:
      impact_selection[impact] = 1
    else:
      impact_selection[impact] += 1
    if damage not in damage_selection:
      damage_selection[damage] = 1
    else:
      damage_selection[damage] += 1
    W += impact_selection[impact]
    Z += damage_selection[damage]
    X *= impact_selection[impact]
    Y *= damage_selection[damage] 

  return max_impact * max_damage * X * Y / (W * Z)
  



if __name__ == '__main__':
    #print("Enter N, H, D, B: ", end = "")
    #N = int(input())
    #H = list(map(int, input().split()))
    #D = list(map(int, input().split()))
    #B = int(input())
    N = 3
    H = [2, 1, 4]
    D = [3, 1, 2]
    B = 4

 
    print(getMaxDamageDealt(N, H, D, B))    
    
    
    #print(getMaxDamageDealt(N, H, D, B))

