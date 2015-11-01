#include <iostream>
#include <vector>
#include <utility>
#include <map>
#include <string>
#include <algorithm>
#include <cstdio>
#include <cmath>

/*4.0 2.0 0.60
4.2	2.1 0.59
3.9 2.0 0.58
4.3 2.1 0.62
4.1 2.2 0.63 */

using namespace std;

int main()
{
	/*int a[5][3] = { 
					{4, 2, 0.6},
					{4.2, 2.1, 0.59},  
					{3.9, 2.0, 0.58},
					{4.3, 2.1, 0.62},
					{4.1, 2.2, 0.63}

				}; */


	float a[100][3];

	/*cout<<"Enter no of rows: "
	int n;
	cin>>n; */

	for(int i=0; i<5; i++)
		for(int j=0; j<3; j++)
			cin>>a[i][j];


//cout<<"hello"<<endl;
	int m=3,n=5;

	float b[3][3]; float sum =0; float avg[3];

	for(int i=0; i<m; i++)
	{
		for(int j=0; j<n; j++)
		{//cout<<"a[j][i]: "<<a[j][i];
			sum += a[j][i];
		}

		//cout<<"sum:"<<sum<<" ";

		float mean  = sum/n;
		avg[i] = mean;

		//cout<<"mean:"<<mean<<" ";

		float var = 0;

		for(int j=0; j<n; j++)
		{
			var += pow(abs(a[j][i] - mean),2);
		}

		b[i][i] = (float)var/(n-1);

		//cout<<b[i][i]<<" ";

		sum = 0;

		//cout<<endl;


	}

	for(int i=0; i<3; i++)
		for(int j=0; j<3; j++)
		{ // although there's no need for this if condition
			if(!(i==j))
			{float prod =0;
				for(int k=0; k<5; k++)
				{
					prod += (avg[i] - a[k][i])*(avg[j] - a[k][j]);
				} 

				//cout<<prod/(n-1)<<endl;

				b[i][j] = prod/(n-1);
			}

		}

	//cout<"The co-variance matrix is:\n";

	for(int i=0; i<3; i++)
	{	for(int j=0; j<3; j++)
			cout<<b[i][j]<<" ";

		cout<<endl; }








	return 0;

}