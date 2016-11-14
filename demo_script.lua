
local Foo = {}

function Foo.foo() -- Static method
    print('Foo !')
end

function Foo:foo2() -- Member
    self.counter = (self.counter and self.counter+1) or 1
    print('Counter value: '..self.counter)
end

function bar() -- Global function
    print('Bar !')
end

return Foo
