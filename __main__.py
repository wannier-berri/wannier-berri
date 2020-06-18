#from .__main import welcome
#from .__version import __version__

#welcome()

from .__main import welcome,print_options
from sys import argv

DO_profile=False


def main():
    if len(argv)>1:
      if argv[1]=='vaspspn':
         from .__vaspspn import main as vaspspn
         vaspspn(argv[1:])
      elif argv[1]=='mmn2uHu':
         from .__mmn2uHu import main as mmn2uHu
         mmn2uHu(argv[1:])
      exit()
    print_options()


if __name__ == "__main__":
#    welcome()
   if DO_profile:
      import cProfile
      cProfile.run('main()')
   else:
      main()



