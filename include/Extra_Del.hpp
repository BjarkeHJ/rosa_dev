#include<iostream>
#include<algorithm>
#include<Eigen/Eigen>

#ifndef EXTRA_DEL_HPP
#define EXTRA_DEL_HPP

class Extra_Del {
public:
	Eigen::MatrixXd rows_ext_V(Eigen::VectorXi ind, Eigen::MatrixXd matrix);
	Eigen::MatrixXd rows_ext_M(Eigen::MatrixXd ind, Eigen::MatrixXd matrix);
	Eigen::MatrixXd cols_ext_V(Eigen::VectorXi ind, Eigen::MatrixXd matrix);
	Eigen::MatrixXd cols_ext_M(Eigen::MatrixXd ind, Eigen::MatrixXd matrix);
	Eigen::MatrixXd rows_del_M(Eigen::MatrixXd ind, Eigen::MatrixXd matrix);
	Eigen::MatrixXd cols_del_M(Eigen::MatrixXd ind, Eigen::MatrixXd matrix);
};

#endif