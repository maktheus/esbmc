//TEST FAILS

#include <string>
#include <cassert>
using namespace std;

int main(){
	string str1, str2;
	str1 = string("Test");
	assert(str1.length() == 4);
	str2 = string(str1, 2);
	assert(str2.length() == 2);
	assert(str2 > str1);
}

