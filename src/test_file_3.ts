interface TestInterface {
    name: string;
    value: number;
}

class TestClass implements TestInterface {
    name = 'test';
    value = 100;
}