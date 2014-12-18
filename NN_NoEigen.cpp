#include <iostream>
#include <string>
#include <sstream>
#include <cmath>

#include <boost/thread.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/foreach.hpp>
#include <boost/progress.hpp>
#include <boost/filesystem.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>
#include <boost/optional.hpp>
#include <boost/random.hpp>

#define	INPUT_LAYER	0
#define	HIDDEN_LAYER	1
#define	OUTPUT_LAYER	2

class NeuronLayer{
public:
	int TypeOfLayer;
	int NumOfNeuron;
	NeuronLayer *PointerOfNext;
	NeuronLayer *PointerOfPrev;
	double *WeightMatrix;
	double *OutputVector;
	double *DeltaVector;
	//Constructer
	NeuronLayer(){
		this->TypeOfLayer = 0;
		this->NumOfNeuron = 0;
		this->PointerOfNext = NULL;
		this->PointerOfPrev = NULL;
		return;
	}
	void Init(int _type,int _num,NeuronLayer *_pp,NeuronLayer *_np){
		this->TypeOfLayer = _type;
		if(_type != OUTPUT_LAYER) _num+=1;
		this->NumOfNeuron = _num;
		this->PointerOfNext = _np;
		this->PointerOfPrev = _pp;
		OutputVector = new double[ _num ];
		for(int i=0;i<_num;i++) OutputVector[i] = 0.0;
		DeltaVector = new double[ _num ];
		for(int i=0;i<_num;i++) DeltaVector[i] = 0.0;
		if(_type != INPUT_LAYER){
			boost::mt19937            gen( static_cast<unsigned long>(time(0)) );
			boost::uniform_smallint<> dst( -10000, 10000 );
			boost::variate_generator< boost::mt19937&, boost::uniform_smallint<> > rand( gen, dst );
			int temp = _num * _pp->NumOfNeuron;
			WeightMatrix = new double[ temp ];
			for(int i=0;i<temp;i++) WeightMatrix[i] = rand() / 10000.0;
		}
		return;
	}
	~NeuronLayer(){
				
	}
	//Forward Compute
	void Compute(double *_input){
		this->OutputVector[0] = 1.0;
		for(int i=0;i<(this->NumOfNeuron);i++) this->OutputVector[i+1] = _input[i];
		Compute();
	}
	void Compute(){
		if( this->TypeOfLayer != INPUT_LAYER ){

			//(this->NumOfNeuron,1) = ( this->NumOfNeuron , this->PointerOfPrev->NumOfNeuron) * ( this->PointerOfPrev->NumOfNeuron , 1)
			for(int i=0;i<(this->NumOfNeuron);i++){
				double temp1 = 0.0;
				for(int j=0;j<(this->NumOfNeuron);j++){
					double temp2 = WeightMatrix[ j + i * this->PointerOfPrev->NumOfNeuron];
					temp2 *= this->PointerOfPrev->OutputVector[j];
					temp1 += temp2;
				}
				OutputVector[i] = temp1;
			}
			//Parallel
			for(int i=0;i<(this->NumOfNeuron);i++)
				OutputVector[i] =  1.0 / ( 1.0 + exp(-OutputVector[i]) );
			if(this->TypeOfLayer != OUTPUT_LAYER) OutputVector[0] = 1.0;
		}
		if( this->TypeOfLayer != OUTPUT_LAYER )
			this->PointerOfNext->Compute();
		return;
	}
	//BackPropagation
	double BackPropagation(double *_teach,double alpha){
		for(int i=0;i<(this->NumOfNeuron);i++){
			//Parallel
			this->DeltaVector[i] = -( _teach[i] - this->OutputVector[i]) * this->OutputVector[i];
			this->DeltaVector[i]*= 1.0 - this->OutputVector[i];
			//Parallel
			for(int j=0;j<(this->PointerOfPrev->NumOfNeuron);j++){
			int index = j + i * (this->PointerOfPrev->NumOfNeuron);
			this->WeightMatrix[index] += -alpha * this->DeltaVector[i] * this->PointerOfPrev->OutputVector[j];
			}
		}
		this->PointerOfPrev->BackPropagation( alpha );
		double e  = 0.0;
		for(int i=0;i<(this->NumOfNeuron);i++)
			e+= pow(_teach[i] - this->OutputVector[i],2.0);
		e *= 0.5;

		return e;
	}
	void BackPropagation(double alpha){
		for(int i=0;i<(this->NumOfNeuron);i++){
			//Parallel
			double sum = 0.0;
			for(int j=0;j<(this->PointerOfNext->NumOfNeuron);j++){
				int index = i + j * this->NumOfNeuron;
				sum += this->PointerOfNext->WeightMatrix[index] * this->PointerOfNext->DeltaVector[j];
			}
			this->DeltaVector[i] = sum * this->OutputVector[i] * (1.0 - this->OutputVector[i]);

			for(int j=0;j<(this->PointerOfPrev->NumOfNeuron);j++){
			int index = j + i * (this->PointerOfPrev->NumOfNeuron);
			this->WeightMatrix[index] += -alpha * this->DeltaVector[i] * this->PointerOfPrev->OutputVector[j];
			}
		}
		if(this->PointerOfPrev->TypeOfLayer != INPUT_LAYER) this->PointerOfPrev->BackPropagation( alpha );
		return;
	}

};

class NeuralNetwork{
private:
public:
	NeuronLayer *layer;
	int NumOfLayer;
	int offset;

	NeuralNetwork(){
		this->layer = NULL;
		this->NumOfLayer = 0;
		this->offset = 0;
	}
	~NeuralNetwork(){
	}

	int Init(int num){
		this->NumOfLayer = num;
		layer = new NeuronLayer[num];
		return 0;
	}

	int Add(int num){
		if( this->offset < this->NumOfLayer ){
			std::cout << "(Add) " << this->offset << std::endl;
			if(this->offset==0)
				layer[this->offset].Init( INPUT_LAYER , num , NULL , &layer[this->offset+1]);
			else if(this->offset==this->NumOfLayer-1)
				layer[this->offset].Init( HIDDEN_LAYER , num , &layer[this->offset-1] , &layer[this->offset+1]);
			else 
				layer[this->offset].Init( OUTPUT_LAYER , num , &layer[this->offset-1] , NULL );
			this->offset++;
			return 0;
		}
		std::cout << "(Err) overflow" << std::endl;
		return -1;
	}

	int Add(int id,int type,int num){
		NeuronLayer *p1,*p2;
		p1 = &layer[id-1];
		p2 = &layer[id+1];
		if(id==0) p1 = NULL;
		if(id==this->NumOfLayer-1) p2 = NULL;
		layer[id].Init( type , num , p1 , p2 );

		return 0;
	}

	int Show(){
		for(int i=0;i<(this->NumOfLayer);i++){
			if(layer[i].NumOfNeuron!=0){
				std::cout << "[" << i << "/" << this->NumOfLayer << "] (" << &layer[i] << ") Prev=" << layer[i].PointerOfPrev;
				std::cout << " Next=" << layer[i].PointerOfNext << std::endl; 
			}else{
				std::cout << "[" << i << "/" << this->NumOfLayer << "] " << std::endl;
			}
		}
		return 0;
	}


	void Compute( double *_input , double *_dest){
		layer[0].Compute( _input );
		if(_dest==NULL) return;
		for(int i=0;i<layer[this->NumOfLayer-1].NumOfNeuron;i++)
		_dest[i] = layer[this->NumOfLayer-1].OutputVector[i];
		return;
	}
	double BackPropagation( double *_teach , double _rate ){
		return layer[this->NumOfLayer-1].BackPropagation( _teach , _rate );
	}

	bool Save(const char* fileName){
		std::vector<double> testVec;
		testVec.push_back( this->NumOfLayer );
		for(int i=1;i<this->NumOfLayer;i++){
			testVec.push_back( layer[i].NumOfNeuron );
			testVec.push_back( layer[i-1].NumOfNeuron );
			for(int j=0;j<layer[i].NumOfNeuron;j++){
				for(int k=0;k<layer[i-1].NumOfNeuron;k++){
					testVec.push_back( layer[i].WeightMatrix[k+j*layer[i-1].NumOfNeuron] );
				}
			}
		}
		std::ofstream ofs(fileName, std::ios::binary);
		if (ofs.fail()) return false;
		ofs.write(reinterpret_cast<const char*>(&testVec[0]), sizeof(double) * testVec.size());
		if (ofs.fail()) return false;
		ofs.flush();
		if (ofs.fail()) return false;
		return true;	
	}

	bool Load(const char* fileName){
		std::ifstream ifs(fileName, std::ios::binary);
		if (ifs.fail()) return false;
		const size_t fileSize = static_cast<size_t>(ifs.seekg(0, std::ios::end).tellg());
		ifs.seekg(0, std::ios::beg);
		if (fileSize > 0 && fileSize % sizeof(double) == 0){
			std::vector<double> testVec(fileSize / sizeof(double));
			ifs.read(reinterpret_cast<char*>(&testVec[0]), fileSize);
			int offset = 0;
			int LayerNum = testVec[ offset++ ];
			for(int i=1;i<LayerNum;i++){
				int row = testVec[ offset++ ];
				int col = testVec[ offset++ ];
				for(int j=0;j<row;j++) for(int k=0;k<col;k++)
					layer[i].WeightMatrix[k+j*col] = testVec[ offset++ ];	
			}
		}else return false;
		return true;
	}

};

//Binary
template<typename T_n> bool SaveMatrix(const char* fileName,int _row,int _col,T_n _mat[]){
	std::vector<T_n> testVec;
	testVec.push_back( _row );
	testVec.push_back( _col );
	for(int i=0;i< _row;i++)
	for(int j=0;j< _col;j++)
	testVec.push_back( _mat[ j + i * _col ] );
	std::ofstream ofs(fileName, std::ios::binary);
	if (ofs.fail()) return false;
	ofs.write(reinterpret_cast<const char*>(&testVec[0]), sizeof(T_n) * testVec.size());
	if (ofs.fail()) return false;
	ofs.flush();
	if (ofs.fail()) return false;
	ofs.close();
	if (ofs.fail()) return false;
	return true;
}

//Binary
template<typename T_n> bool LoadMatrix(const char* fileName ,int *info , T_n _mat[]){
	std::ifstream ifs(fileName, std::ios::binary);
	if (ifs.fail()) return false;
	const size_t fileSize = static_cast<size_t>(ifs.seekg(0, std::ios::end).tellg());
	ifs.seekg(0, std::ios::beg);
	std::vector<T_n> testVec(fileSize / sizeof(T_n));
	ifs.read(reinterpret_cast<char*>(&testVec[0]), fileSize);
	ifs.close();
	int offset = 0;
	int _row = testVec[ offset++ ];
	int _col = testVec[ offset++ ];
	//std::cout << "row=" << _row << std::endl;
	//std::cout << "col=" << _col << std::endl;
	if(info!=NULL){
	info[0] = _row;
	info[1] = _col;
	}
	for(int i=0;i<_row;i++)
		for(int j=0;j<_col;j++)
			_mat[ j + i * _col] = testVec[ offset++ ];
	return true;
}

template<typename T_n> void  LoadCsv(std::string filename,int info[],T_n mat[]){
	std::ifstream ifs( filename );
	std::string str;
	if( !ifs ) {
		std::cout << "Error:Input data file not found" << std::endl;
		return;
	}
	int cnt=0;
	int row = 0;
	int col = 0;
	while( std::getline( ifs, str ) ){
		std::string token;
		std::istringstream stream( str );
		int TempCol = 0;
		while( std::getline( stream, token, ',' ) ) {
			std::stringstream ss;
			double temp;
			ss << token;
			ss >> temp;
			mat[cnt] = temp;
			TempCol++;
			cnt++;
		}
		if( col < TempCol ) col = TempCol;
		row++;
	}
	if(info!=NULL){
		info[0] = row;
		info[1] = col;
	}
	return;
}


void MakeData(){

	double Input[] = 
	{1,1,1,1,1,
	1,0,0,0,1,
	1,0,0,0,1,
	1,0,0,0,1,
	1,1,1,1,1,
	//1
	0,0,1,0,0,
	0,0,1,0,0,
	0,0,1,0,0,
	0,0,1,0,0,
	0,0,1,0,0,
	//2
	1,1,1,1,1,
	0,0,0,0,1,
	1,1,1,1,1,
	1,0,0,0,0,
	1,1,1,1,1,
	//3
	1,1,1,1,1,
	0,0,0,0,1,
	1,1,1,1,1,
	0,0,0,0,1,
	1,1,1,1,1,
	//4
	1,0,0,1,0,
	1,0,0,1,0,
	1,1,1,1,1,
	0,0,0,1,0,
	0,0,0,1,0,
	//5
	1,1,1,1,1,
	1,0,0,0,0,
	1,1,1,1,1,
	0,0,0,0,1,
	1,1,1,1,1,
	//6
	1,0,0,0,0,
	1,0,0,0,0,
	1,1,1,1,1,
	1,0,0,0,1,
	1,1,1,1,1,
	//7
	1,1,1,1,1,
	1,0,0,0,1,
	1,0,0,0,1,
	0,0,0,0,1,
	0,0,0,0,1,
	//8
	1,1,1,1,1,
	1,0,0,0,1,
	1,1,1,1,1,
	1,0,0,0,1,
	1,1,1,1,1,
	//9
	1,1,1,1,1,
	1,0,0,0,1,
	1,1,1,1,1,
	0,0,0,0,1,
	0,0,0,0,1};

	double Output[] = {
	1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
	0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
	0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
	0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,
	0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,
	0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,
	0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,
	0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,
	0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,
	0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0
	};

	SaveMatrix<double>("A.bin",10,25,Input);
	SaveMatrix<double>("B.bin",10,10,Output);

	return;
}


int main(int argc,char *argv[]){

	int NumOfSample = 10;
	int DimOfInput = 25;
	int DimOfOutput = 10;
	double *Input = new double[NumOfSample * DimOfInput];
	double *Output = new double[NumOfSample * DimOfOutput];

	NeuralNetwork nn1;
	nn1.Init(3);
	nn1.Add(0,INPUT_LAYER,DimOfInput);
	nn1.Add(1,HIDDEN_LAYER,20);
	nn1.Add(2,OUTPUT_LAYER,DimOfOutput);
	nn1.Show();
	MakeData();

	int info[2];
	LoadMatrix<double>("A.bin",info,Input);
	std::cout << "[A] " << info[0] << " " << info[1] << std::endl;
	LoadMatrix<double>("B.bin",info,Output);
	std::cout << "[B] " << info[0] << " " << info[1] << std::endl;

	std::cout << "Loading Neural Network..." << std::endl;

	if( nn1.Load("N.bin") ){
		std::cout << "Load Passed." << std::endl;
	}else{
		std::cout << "Load Failed." << std::endl;
		int Iter = 0;
		while(1){
			//Back-Propagation
			double e = 0.0;
			for(int i=0;i<NumOfSample;i++){
				nn1.Compute( Input + DimOfInput * i , NULL );
				e+=nn1.BackPropagation( Output + DimOfOutput * i , 0.5 );
			}
			std::cout << "[" << Iter << "] e=" << e << std::endl;
			//Interval Save Function
			if( Iter!=0 && Iter%100 == 0){
				if( nn1.Save("N.bin") ){
					std::cout << "Interval Save Passed." << std::endl;
				}else{
					std::cout << "Interval Save Failed." << std::endl;
				}
			}
			//Convergence Check
			if( e < 0.001){
				break;
			}
			Iter++;
		}
		if( nn1.Save("N.bin") ){
			std::cout << "Save Passed." << std::endl;
		}else{
			std::cout << "Save Failed." << std::endl;
		}
	}

	double Temp[DimOfOutput];
	for(int i=0;i<NumOfSample;i++){
		nn1.Compute( Input + DimOfInput * i , Temp );
		std::cout << "[" << i << "] ";
		for(int j=0;j<DimOfOutput;j++)
			std::cout << Temp[j] << ",";
		std::cout << std::endl;
	}
	
	delete[] Input;
	delete[] Output;

	return 0;
}
