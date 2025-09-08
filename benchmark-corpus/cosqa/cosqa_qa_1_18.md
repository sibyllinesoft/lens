# Q&A Pair 1 - Variant 18

## Question
What is the difference between == and === in JavaScript?

## Answer
The difference between == and === in JavaScript:

== (Equality Operator):
- Performs type coercion before comparison
- Converts operands to same type then compares
- Examples:
  5 == "5" // true (string "5" converted to number)
  true == 1 // true (boolean converted to number)
  null == undefined // true

=== (Strict Equality Operator):
- No type coercion
- Compares both value and type
- Examples:
  5 === "5" // false (different types)
  true === 1 // false (different types)
  null === undefined // false

Best practice: Use === to avoid unexpected type conversions.

## Additional Context
This is a commonly asked programming question in interviews.