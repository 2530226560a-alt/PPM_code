#pragma once
#include<vector>
#include <iostream>
#include<string>
using namespace std;

class petrick
{
private:
	vector<string> vec;
	vector<string> result;
	double * prob;

public:
	petrick(vector<string> inputVec, double* input_prob)
	{
		vec = inputVec;
		prob = input_prob;
	}

	void show(int flag, vector<string> v)
	{
		if (v.size() > 0)
		{
			if (flag == 0)
				cout << " P =\t\b\b\b\b";
			else
				cout << " \t\b\b\b\b";
			int cc = 0;
			for (int k = 0; k < v.size(); k++)
			{
				cout << "(";
				for (int i = 0; i < v[k].size(); i++)
				{
					cout << (int)v[k][i];
					cout << "v";
				}
				cout << "\b)";
				cc++;
				if (cc == 5)
				{
					cout << endl << "\t\b\b\b\b";
					cc = 0;
				}
			}
			cout << endl;
		}
		else
		{
			cout << endl << " P = empty" << endl;
		}

	}

	void out()
	{
		for (int i = 0; i < (int)vec.size(); i++)
			cout << vec[i] << endl;
	}

	void showResult()
	{
		for (int i = 0; i < (int)result.size(); i++)
			cout << result[i] << endl;
	}

	vector<string> multi(vector<string> A, vector<string> B)
	{
		string temp;
		vector<string> C;
		for (int i = 0; i < A.size(); i++)
			for (int j = 0; j < B.size(); j++)
			{
				C.push_back(A[i] + B[j]);
			}

		bool flag;
		for (int k = 0; k < C.size(); k++)
		{
			flag = 1;
			while (flag == 1)
			{
				flag = 0;
				for (int i = 0; i < C[k].size(); i++)
				{
					for (int j = 0; j < C[k].size(); j++)
						if ((C[k][i] == C[k][j]) && (i != j))
						{
							C[k].erase(C[k].begin() + j);
							flag = 1;
							break;
						}
					if (flag == 1)
						break;
				}

			}
		}

		int count = 0;
		for (int k = 0; k < C.size(); k++)
			for (int kk = 0; kk < C.size(); kk++)
				if (k != kk)
				{
					if (C[kk].size() < C[k].size() || (C[kk].size() == 0) || (C[k].size() == 0))
						continue;
					else
					{
						count = 0;
						for (int i = 0; i < C[k].size(); i++)
							for (int ii = 0; ii < C[kk].size(); ii++)
								if (C[k][i] == C[kk][ii])
									count = count + 1;
						if (count == C[k].size())
							C[kk] = { " " };
					}
				}

		for (int k = 0; k < C.size(); k++)
			if (C[k] == " ")
			{
				C.erase(C.begin() + k);
				k--;
			}


		return C;
	}

	bool contain(string A, string B)
	{
		bool flag;
		for (int i = 0; i < A.size(); i++)
		{
			flag = 0;
			for (int j = 0; j < B.size(); j++)
			{
				if (A[i] == B[j])
				{
					flag = 1;
					break;
				}
			}
			if (flag == 0)
				return 0;
		}
		return 1;
	}

	vector<string> run()
	{
		cout << " Petrick function:" << endl << endl;
		show(0, vec);
		vector<string> C;
		vector<string> D;
		C.push_back("");
		char temp;
		for (int k = 0; k < vec.size(); k++)
		{
			if (vec[k].size() == 1)
			{
				//C.push_back({ vec[k][0] });
				C[0] = C[0] + vec[k][0];
				temp = vec[k][0];
				for (int l = 0; l < vec.size(); l++)
				{
					for (int m = 0; m < vec[l].size(); m++)
					{
						if (temp == vec[l][m])
						{
							vec.erase(vec.begin() + l);
							l--;
							k = -1;
							m = -1;
						}
						if (m == -1)
							break;
					}

				}
			}
		}

		cout << endl << " First step -- find essential candidates:" << endl << endl;
		if (C[0].size() > 0)
		{
			cout << " P =\t\b\b\b\b";
			for (int i = 0; i < C[0].size(); i++)
				cout << "(" << (int)C[0][i] << ")";
			cout << endl;
			if (vec.size() > 0)
				show(1, vec);
		}
		else
			show(0, vec);
		cout << endl;


		if (vec.size() > 0)
		{
			cout << endl << " Second step -- find groups:" << endl;

			int maxsize = 0;
			for (int k = 0; k < vec.size(); k++)
			{
				if (maxsize < vec[k].size())
					maxsize = vec[k].size();
			}

			bool flag = 0;
			for (int i = 2; i <= maxsize; i++)
			{
				for (int k = 0; k < vec.size(); k++)
				{
					flag = 0;
					if (vec[k].size() == i)
					{
						D.push_back({ vec[k] });
						vec.erase(vec.begin() + k);
						k = 0;
						flag = 1;
					}

					if (flag == 1)
					{
						for (int l = 0; l < vec.size(); l++)
						{
							if (contain(D[D.size() - 1], vec[l]))
							{
								vec.erase(vec.begin() + l);
								l--;
							}
						}

						//show
						cout << endl;
						if (C[0].size() > 0)
						{
							cout << " P =\t\b\b\b\b";
							for (int i = 0; i < C[0].size(); i++)
								cout << "(" << (int)C[0][i] << ")";
							cout << endl;
							if (D.size() > 0)
								show(1, D);
							if (vec.size() > 0)
								show(1, vec);
						}
						else if (D.size() > 0)
						{
							show(0, D);
							if (vec.size() > 0)
								show(1, vec);
						}
						else
							show(0, vec);
						cout << endl;

					}
				}
			}
		}

		if (D.size()>0 || vec.size() > 0)
		{
			cout << endl << "After second step -- find group:" << endl;
			if (C[0].size() > 0)
			{
				cout << "P =\t\b\b\b\b";
				for (int i = 0; i < C[0].size(); i++)
					cout << "(" << (int)C[0][i] << ")";
				cout << endl;
				if (D.size()>0)
					show(1, D);
				if (vec.size() > 0)
					show(1, vec);
			}
			else if (D.size() > 0)
			{
				show(0, D);
				if (vec.size() > 0)
					show(1, vec);
			}
			else
				show(0, vec);
			cout << endl;

			for (int i = 0; i < D.size(); i++)
				vec.push_back(D[i]);
		}

		if (D.size() > 0)
		{
			for (int i = 0; i < D.size(); i++)
				vec.push_back(D[i]);
		}

		if (vec.size() > 0)
		{
			cout << " Third step -- multiply out:" << endl << endl;
			vector<string> A;
			for (int i = 0; i < vec[0].size(); i++)
				A.push_back({ vec[0][i] });

			if (vec.size() > 1)
			{
				vector<string> B;
				for (int k = 1; k < vec.size(); k++)
				{
					B.clear();
					for (int i = 0; i < vec[k].size(); i++)
						B.push_back({ vec[k][i] });
					A = multi(A, B);

					//show
					cout << endl;
					if (C[0].size() > 0)
					{
						cout << " P =\t\b\b\b\b";
						for (int i = 0; i < C[0].size(); i++)
							cout << "(" << (int)C[0][i] << ")";
						cout << endl;
						cout << "\t\b\b\b\b";
						cout << "[";
						for (int l = 0; l < A.size(); l++)
						{
							if (l != 0)
								cout << " v ";
							cout << "(";
							for (int i = 0; i < A[l].size(); i++)
							{
								cout << "(";
								cout << (int)A[l][i];
								cout << ")";
							}
							cout << "\b)";
						}
						cout << "]";
						cout << endl;

						if (vec.size() - 1 - k > 0)
						{

							cout << " \t\b\b\b\b";
							int cc = 0;
							for (int m = k + 1; m < vec.size(); m++)
							{
								cout << " (";
								for (int i = 0; i < vec[m].size(); i++)
								{
									cout << (int)vec[m][i];
									cout << " v";
								}
								cout << " \b)";
								cc++;
								if (cc == 5)
								{
									cout << endl << " \t\b\b\b\b";
									cc = 0;
								}
							}
							cout << endl;
						}
					}
					cout << endl;

				}
			}

			double tempprob = 1;
			double maxprob = 0;
			int idx;
			vector<string> E;
			cout<<"A.size(): " << A.size() << endl;
			if (A.size() >= 1)
			{
				cout << " Fourth step -- compare likelihood:" << endl << endl;

				for (int k = 0; k < A.size(); k++)
				{
					tempprob = 1;
					for (int i = 0; i < A[k].size(); i++)
					{
						tempprob *= prob[(int)A[k][i]];
						cout << " " << (int)A[k][i] << " ";
					}
					cout << " Joint Likelihood:" << tempprob << endl;
					if (tempprob > maxprob)
					{
						maxprob = tempprob;
						idx = k;
					}
				}

				E.push_back(A[idx]);

				cout << endl << " Win: ";
				for (int i = 0; i < E[0].size(); i++)
				{
					cout << (int)E[0][i] << " ";
				}
				cout << endl;
			}

			if (C[0].size() > 0)
				E = multi(C, E);

			result = E;

			cout << endl;
			if (result.size() > 0)
			{
				cout << " P =\t\b\b\b\b";
				int cc = 0;
				for (int k = 0; k < result.size(); k++)
				{
					if (k != 0)
						cout << " + ";
					cout << "(";
					for (int i = 0; i < result[k].size(); i++)
					{
						cout << "(";
						cout << (int)result[k][i];
						cout << ")";
					}
					cout << "\b)";

					cc++;
					if (cc == 5)
					{
						cout << endl << " \t\b\b\b\b";
						cc = 0;
					}
				}
				cout << endl;
			}

			cout << endl;
			cout << " Function minimisation is finished." << endl;
			cout << "result!" << endl;
			cout << "result.size(): " << result.size() << endl;
			return result;
		}
		else
		{
			cout << " Function minimisation is finished." << endl;
			cout << "C!" << endl;
			return C;
		}
	}

};