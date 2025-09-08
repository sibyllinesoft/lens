# Q&A Pair 2 - Variant 29

## Question
How to handle exceptions in Java?

## Answer
Exception handling in Java uses try-catch-finally blocks:

Basic syntax:
try {
    // Code that might throw exception
    int result = 10 / 0;
} catch (ArithmeticException e) {
    // Handle specific exception
    System.out.println("Division by zero: " + e.getMessage());
} catch (Exception e) {
    // Handle general exception
    System.out.println("General error: " + e.getMessage());
} finally {
    // Always executes (cleanup code)
    System.out.println("Cleanup operations");
}

Key concepts:
- Checked exceptions: Must be caught or declared (IOException)
- Unchecked exceptions: Runtime exceptions (NullPointerException)
- Custom exceptions: Extend Exception or RuntimeException classes
- throws keyword: Declare exceptions in method signature

## Examples
Practice with similar problems to master this concept.