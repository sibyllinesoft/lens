# Q&A Pair 0 - Variant 89

## Question
How to sort a list in Python?

## Answer
There are several ways to sort a list in Python:

1. Using sort() method (modifies original list):
   my_list = [3, 1, 4, 1, 5]
   my_list.sort()  # [1, 1, 3, 4, 5]

2. Using sorted() function (returns new list):
   my_list = [3, 1, 4, 1, 5]
   sorted_list = sorted(my_list)  # [1, 1, 3, 4, 5]

3. Custom sorting with key parameter:
   students = [('Alice', 85), ('Bob', 90), ('Charlie', 78)]
   students.sort(key=lambda x: x[1])  # Sort by grade

4. Reverse sorting:
   my_list.sort(reverse=True)  # Descending order

## Examples
Practice with similar problems to master this concept.