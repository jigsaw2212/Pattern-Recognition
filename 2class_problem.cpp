#include <iostream>
#include <vector>
#include <utility>
#include <map>
#include <string>
#include <algorithm>
#include <cstdio>
#include <cmath>
#include <string.h>
#include <climits>
#include <cmath>

using namespace std;

int main()
{
	vector<int> girls;
	vector<int> boys;

	map<int,int> g;
	map<int,int> b;

	int n1,n2,num,l,r;

	cout<<"Enter no of boys: ";
	cin>>n1;

	cout<<"\nEnter no of girls: ";
	cin>>n2;

//n1 - boys
//n2 - girls

	cout<<"\nEnter heights of boys:- \n";

	for(int i=0; i<n1; i++)
	{
		cin>>num;

		b[num]++;
		boys.push_back(num);

	}

	cout<<"\nEnter heights of girls:- \n";

	for(int i=0; i<n2; i++)
	{
		cin>>num;

		g[num]++;
		girls.push_back(num);

	}

	int val;

	cout<<"\nEnter val: ";
	cin>>val;

	float pXgW1,pXgW2, pw1, pw2;

	pXgW1 = g[val]/(n2 * 1.0);

	pXgW2 = b[val]/(n1 * 1.0);

	pw1 = n1/(float)(n1+n2);
	pw2 = n2/(float)(n1+n2);

	if (pXgW1*pw1 > pXgW2*pw2)
	{
		cout<<"\nBelongs to class Girls\n";
	}

	else if (pXgW1*pw1 < pXgW2*pw2)
	{
		cout<<"\nBelongs to class boys\n";
	}

	else
	{
		if(pw1 > pw2)
			cout<<"\nBelongs to class Boys\n";
		else if(pw1 < pw2)
			cout<<"\nBelongs to class Girls\n";
		else
			cout<<"\nCan belong to any class\n";
	}

	return 0;
}

/*
Data:-
Girls:-
138  
142  
137 
138 
151 
147 
151 
151 
137 

Boys:-
158
157
158
152
151
153
154
137 */
