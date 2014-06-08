#include <iostream>
#include <fstream>

int main()
{
  char c;
  std::ifstream f_in("mt_gambier_512.ubc");
  std::ofstream f_out("mt_gambier_512.data");

  for (unsigned int x=0;x<512;++x)
  for (unsigned int y=0;y<512;++y)
  for (unsigned int z=0;z<512;++z)
    {
      char c;
      f_in >> c;
      //      std::cout << (int)c << " ";
      
      if (c==1)
	c=0xff;

      if (z<300 && x<300)
	c=0;
      
	
      f_out << c;
    }
  

}
