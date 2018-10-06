import tensorConnect5 as agent

global agent

def main():
    command = "r"
    global agent
    agent = agent.Agent(epsilon=0.9)
    while True:
        try:
            agent.train()
        except KeyboardInterrupt:
            while True:
                command = input("Enter a command: (R)esume -- (C)hange Value -- (S)ave -- (Q)uit:  ").lower().strip()
                if command in {"r", "resume"}:
                    break
                operation(command)

def operation(command:str):
    if command in {"c", "change", "change value"}:
        change()
    elif command in {"s", "save"}:
        agent.save_model()
    elif command in {"q", "quit"}:
        quit()

def change():
    global agent
    list_attr = [key for key in agent.__dict__.keys()]
    while True:
        attr = input("Which attribute would you like to change? {}\n ('q' to quit)\n".format(list_attr))
        if attr.lower() in {"q", "quit"}:
            break
        
        if attr in list_attr:
            value = input("Please enter the new value to set to:  ")
            try:
                value = type(agent.__dict__[attr])(value)
                agent.__dict__[attr] = value
                print("Changed!")
                break
            except ValueError:
                print("Invalid value type; Please try again (must be {})".format(type(agent.__dict__[attr])))
        else:
            print("Invalid attribute name; Please try again (case sensitive)")


if __name__ == "__main__":
    main()

