# Arc_Welding  

- 아크 용접 데이터를 이용한 길이 추종 프로젝트  


## Description  

- data 파일 불러오기
- 3차원 점들을 지나는 평면 추종(ransac) + 해당 평면에 좌표 사영
- 시작과 끝점 기준 직선 ransac
- 두 직선에 대한 교점(혹은 가장 가까운 점과 거리)
- 

### 주의사항

- 가상환경 설치 시, python 버전은 3.10으로(scikit-learn 버전 호환성)


(추가)  
- 데이터 수집, 애니메이션 라이브러리 추가,   

(최근 진행 중)  
- Polynomial RANSAC 적용 중; 2d(curve), 3d(curve, plane)