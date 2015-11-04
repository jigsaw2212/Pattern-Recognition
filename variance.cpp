#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

int main()
{
	int n; float num;

	cout<<"Enter size:";

	cin>>n;


	vector<float> a;

	cout<<"\nEnter numbers:-";

	float sum=0;

	for(int i=0; i<n; i++)
		{cin>>num; a.push_back(num); 
			sum+=num; }


	float mean= (sum/n);
	

	cout<<"\nMean: "<<mean<<endl;

	sum=0;

	for(int i=0; i<n; i++)
	{
		sum += pow((abs(a[i] - mean)),2);
	}

	cout<<"Variance: "<<(sum/n)<<endl;

	cout<<"Standard Deviation: "<<sqrt((sum/n))<<endl;


	return 0;
}