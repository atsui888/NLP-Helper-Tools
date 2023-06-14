
from pathlib import Path

state_dict_file_name = 'model_state_dict.pth'
cwd = Path.cwd()
state_dict_path_file = cwd / state_dict_file_name


def display_menu():
    print('*' * 50)
    print('Menu')
    print('*' * 50)

    if Path.exists(state_dict_path_file):
        print("Saved model state_dict found.")
        menu_str = """
        (r). Reuse saved model
        (t). Train a new model
        (e). Exit
        """
    else:
        print("No saved model found.")
        menu_str = """        
        (t). Train a new model
        (e). Exit
        """
    print(menu_str)


def get_user_menu_choice():
    while True:
        display_menu()
        try:
            choice = str(input('What do you want to do: ')).lower()
            if choice in ['r', 't', 'e']:
                if choice == 'r':
                    print("You have chosen to reuse the saved data.")
                elif choice == 't':
                    msg = "You have chosen to train a new model, please confirm by typing 'fresh'. "
                    choice = 't' if str(input(msg)).lower() == 'fresh' else 'r'
                    if choice == 'r' and not Path.exists(state_dict_path_file):
                        print("You do not wish to train a new model and there is no saved model.")
                        choice = 'e'
                    return choice
                elif choice == 'e':
                    print("Exiting program")

                return choice
            else:
                print("Invalid Selection.")
        except TypeError:
            print("Invalid data type")


if __name__ == "__main__":

    user_choice = get_user_menu_choice()
    if user_choice == 'r':
        print('Loading Saved Model')
    elif user_choice == 't':
        print('Starting a New training run.')
    else:
        print('Shutting down program. Have a good day!')

