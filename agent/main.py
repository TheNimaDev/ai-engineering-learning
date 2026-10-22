
class Agent:
    def __init__(self):
        pass

    def chat(self):
        while True:
            query = input(
                "Enter some text (type q to terminate): "
            ).strip()

            if query.lower() == "q":
                print("Good Luck")
                break

            if not query:
                print("Please enter some text.")
                continue



def main():
    agent = Agent()

    try:
        agent.chat()
    except KeyboardInterrupt:
        print("\n\nChat interrupted by user.")


if __name__ == "__main__":
    main()
