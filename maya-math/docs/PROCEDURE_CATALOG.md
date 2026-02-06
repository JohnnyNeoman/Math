# Maya MEL Procedure Catalog

> Comprehensive catalog of MEL procedures organized for SST integration
> Generated from legacy scripts - annotated for future Python translation

---

## Summary

- **Total Procedures**: 2308
- **Unique Procedures**: 749
- **Files Processed**: 68
- **Duplicate Names**: 1559

### By Category

| Category | Count | SST Layer | Description |
|----------|-------|-----------|-------------|
| **circle** | 128 | conformal | Circle creation, 3-point circles, packing |
| **tangent** | 106 | conformal | Point-to-circle tangents, tangent circles |
| **sketch** | 118 | conformal/spectral | CAM-based curve sketching, projection |
| **curve** | 389 | conformal | NURBS curves, arc operations |
| **matrix** | 160 | affine | Matrix operations, rotations |
| **polygon** | 149 | affine | Polygon operations, faces, edges |
| **array** | 565 | affine | Array manipulation, sorting, conversion |
| **utility** | 693 | affine | General utilities, selection, cleanup |

---

## CIRCLE Procedures

**SST Layer**: conformal
**Description**: Circle creation, 3-point circles, packing

| Procedure | Return | Parameters | Source |
|-----------|--------|------------|--------|
| `AverageANdCLoseSketchFittingCircle` | string | `string $curveItemC` | Maya_MEL_Proc_Scripts\MuchBett |
| `boundingCircleRadius` | float | `int $n, float $x` | Maya_MEL_Proc_Scripts\ALL MEL  |
| `Circle3PtsM` | float [] | `float $p1[], float $p2[], float $p3[]` | Maya_MEL_Proc_Scripts\RADIAL_S |
| `Circle3PtZB` | void | `` | Maya_MEL_Proc_Scripts\fixedITC |
| `Circle3PtZFloats` | float [] | `float $p1[], float $p2[], float $p3[]` | Maya_MEL_Proc_Scripts\RADIAL_S |
| `Circle3PtZFloatsI` | float [] | `float $p1[], float $p2[], float $p3[]` | Maya_MEL_Proc_Scripts\RADIAL_S |
| `CircleBetweenCircle` | void | `` | Maya_MEL_Proc_Scripts\TANGENTZ |
| `CircleFromCurve` | void | `` | Maya_MEL_Proc_Scripts\True Tan |
| `CircleFromCurveN` | void | `int $X, int $Y, int $Z` | Maya_MEL_Proc_Scripts\TANGENTZ |
| `CircleFromCurveRadiusZYX` | void | `int $X, int $Y, int $Z` | Maya_MEL_Proc_Scripts\TANGENTZ |
| `CircleFromCurveT` | void | `` | Maya_MEL_Proc_Scripts\TANGENTZ |
| `CircleFromCurveZYX` | void | `int $X, int $Y, int $Z` | Maya_MEL_Proc_Scripts\TANGENTZ |
| `Circles` | string | `float $Point[], float $FloatNum` | Maya_MEL_Proc_Scripts\True Tan |
| `Circles_Direction` | string | `float $Point[], float $FloatDir[], float...` | Maya_MEL_Proc_Scripts\RADIAL_S |
| `CIRCLESCRIPTZ` | string[] | `string $ObjectCurve[]` | Maya_MEL_Proc_Scripts\Circle P |
| `CirclesRadiiPos` | float [] | `string $CirObjects[], vector $PosAB[]` | Maya_MEL_Proc_Scripts\TANGENTZ |
| `CirclesRadius` | float | `string $selectedObjects[]` | Maya_MEL_Proc_Scripts\TANGENTZ |
| `CirclesRadiusDirection` | float [] | `vector $directionAB[], vector $PosAB[]` | Maya_MEL_Proc_Scripts\TANGENTZ |
| `CircleSRelationCircle` | void | `` | Maya_MEL_Proc_Scripts\True Tan |
| `CircNormal` | float [] | `string $eachCirV` | Maya_MEL_Proc_Scripts\TANGENTZ |
| `CreateCircleBetweenCircle` | void | `` | Maya_MEL_Proc_Scripts\True Tan |
| `CreateCircleINpolyFaces` | void | `` | Maya_MEL_Proc_Scripts\Circle P |
| `CurvatureIsCircle` | float [] | `string $curvesZ, int $NumberofSteps, flo...` | Maya_MEL_Proc_Scripts\MuchBett |
| `CurvatureIsCircleData` | vector [] | `string $curvesZ, int $NumberofSteps, flo...` | Maya_MEL_Proc_Scripts\MuchBett |
| `Eval3DCircleStereo` | string [] | `int $NewArrayBEllipZ[] ,vector $EmptyVec...` | Maya_MEL_Proc_Scripts\MuchBett |
| `IntersectTwoCircles` | void | `` | Maya_MEL_Proc_Scripts\ThreeCir |
| `IsCircle` | int | `` | Maya_MEL_Proc_Scripts\Circle P |
| `IScircleTF` | string | `string $EachCrvX, int $TFCc` | Maya_MEL_Proc_Scripts\MuchBett |
| `IsPointArray_in_ThreePointCircle_Global` | int | `int $threeIndex[]` | Maya_MEL_Proc_Scripts\ThreePoi |
| `MakeCIRCLE` | string[] | `string $ObjectCurve[]` | Maya_MEL_Proc_Scripts\Circle P |
| `PointInCircle` | int[] | `vector $pos[], vector $posA, float $radi...` | Maya_MEL_Proc_Scripts\fixedITC |
| `PointToCircleTangents` | vector [] | `float $CircleRadiusA, float $worldPosA[]...` | Maya_MEL_Proc_Scripts\TANGENTZ |
| `TangentCircleBetweenCircle` | void | `` | Maya_MEL_Proc_Scripts\TANGENTZ |
| `TangentCircles` | void | `` | Maya_MEL_Proc_Scripts\TANGENTZ |
| `TangentCirclesAtand` | void | `string $MathCommand` | Maya_MEL_Proc_Scripts\TANGENTZ |
| `TangentPointCircles` | void | `` | Maya_MEL_Proc_Scripts\TANGENTZ |
| `TangentPointCircles2` | void | `` | Maya_MEL_Proc_Scripts\TANGENTZ |
| `TangentPointCirclesVec2` | vector [] | `` | Maya_MEL_Proc_Scripts\TANGENTZ |
| `ThreeCirclesMakeMiddleCi` | void | `` | Maya_MEL_Proc_Scripts\ThreeCir |

---

## TANGENT Procedures

**SST Layer**: conformal
**Description**: Point-to-circle tangents, tangent circles

| Procedure | Return | Parameters | Source |
|-----------|--------|------------|--------|
| `AdvancedCurveMODprojectTAN` | void | `string $INarrayA [], int $plainNum` | Maya_MEL_Proc_Scripts\Circle P |
| `ArrayDistancesVecTofloat` | float [] | `vector $allVec[], float $point[]` | Maya_MEL_Proc_Scripts\RADIAL_S |
| `attachCurveTangent` | string | `int $doAttach` | Maya_MEL_Proc_Scripts\Workin T |
| `distance2PtS` | float | `float $p1[], float $p2[]` | Maya_MEL_Proc_Scripts\RADIAL_S |
| `distanceBetween` | float | `float $loc1[], float  $loc2[]` | Maya_MEL_Proc_Scripts\Circle P |
| `DistanceSortStereoCrv` | float [] | `float $distToCurveE_A[], vector $LocCurv...` | Maya_MEL_Proc_Scripts\MuchBett |
| `edgeDistance` | float | `string $edge, string $vtx` | Maya_MEL_Proc_Scripts\ALL MEL  |
| `findParamAtDistance` | float | `string $curve, string $arcLD, float $dis...` | Maya_MEL_Proc_Scripts\matrixRo |
| `GetAngleEulerBetween` | float [] | `float $pointA[], float $pointB[]` | Maya_MEL_Proc_Scripts\Circle P |
| `getDistance` | float | `float  $pointAZ1[], float $pointAZ2[]` | Maya_MEL_Proc_Scripts\Circle P |
| `GetDistance` | int | `string $objectFirst, string  $objectSeco...` | Maya_MEL_Proc_Scripts\Circle P |
| `GetDistanceBetweenCurveEnds` | float | `string  $CurveItem[]` | Maya_MEL_Proc_Scripts\Circle P |
| `GetDistanceFLOAT` | float | `string $objectFirst,  string $objectSeco...` | Maya_MEL_Proc_Scripts\Circle P |
| `GetDistancePointPositionFLOAT` | float | `string $objectFirst, string $objectSecon...` | Maya_MEL_Proc_Scripts\Circle P |
| `InsideRectangle` | int [] | `string $is,string $AllOther[]` | Maya_MEL_Proc_Scripts\THE_one_ |
| `KILLtanCurveRUNautoboundry` | void | `` | Maya_MEL_Proc_Scripts\Circle P |
| `MovePointDirectionAndDistance` | float [] | `float $Direction[], float $Distance, flo...` | Maya_MEL_Proc_Scripts\RADIAL_S |
| `plotEquidistantLocatorsOnCurve` | void | `string $curve, int $count` | Maya_MEL_Proc_Scripts\matrixRo |
| `PointDirTang2LineVec` | vector | `float $DirectionLineF[], float $PointOnL...` | Maya_MEL_Proc_Scripts\MuchBett |
| `PointDistance_Plane` | float | `vector $Vec_Array, vector $AXIS_XZY[],ve...` | Maya_MEL_Proc_Scripts\RADIAL_S |
| `PointsGetDistanceFLOAT` | float | `float  $pointAZ1[], float $pointAZ2[]` | Maya_MEL_Proc_Scripts\Circle P |
| `RUNprojectTAN` | void | `` | Maya_MEL_Proc_Scripts\Circle P |
| `TangentPointCirVectors` | vector [] | `float $ObjsCirclesRad[], vector $PosABs[...` | Maya_MEL_Proc_Scripts\TANGENTZ |
| `TANGENTSS` | vector [] | `` | Maya_MEL_Proc_Scripts\TANGENTS |
| `TRIGGER_RUNjobNumTAN` | void | `` | Maya_MEL_Proc_Scripts\Circle P |
| `zenEdgeDistance` | int[] | `string $vert,string $verts[]` | Maya_MEL_Proc_Scripts\full aut |

---

## SKETCH Procedures

**SST Layer**: conformal/spectral
**Description**: CAM-based curve sketching, projection

| Procedure | Return | Parameters | Source |
|-----------|--------|------------|--------|
| `AdvancedCurveMODprojectOnSurface` | void | `string  $AllIntersectCurveSA[], string $...` | Maya_MEL_Proc_Scripts\Circle P |
| `CatchMoveZCURVECAM` | void | `string $EdgeCurveZ2[],  string $CamConeL...` | Maya_MEL_Proc_Scripts\Circle P |
| `CatchMoveZCURVECAM2010` | void | `string $EdgeCurveZa[], string $CamConeLo...` | Maya_MEL_Proc_Scripts\RADIAL_S |
| `ConvertPolyPlaneIntoNurbSurfaceZ` | string[] | `string $PolygonNew[]` | Maya_MEL_Proc_Scripts\Circle P |
| `CreateCAMforIntCurveScript` | string[] | `` | Maya_MEL_Proc_Scripts\Circle P |
| `CurvesToPlane` | void | `string $CurveItemZ[]` | Maya_MEL_Proc_Scripts\MuchBett |
| `DoesCurveBBcrossPlane` | int | `string $CrvS` | Maya_MEL_Proc_Scripts\THE_one_ |
| `EachPointToCameraPlane` | void | `` | Maya_MEL_Proc_Scripts\Workin T |
| `EVALCamScripts` | void | `` | Maya_MEL_Proc_Scripts\Circle P |
| `EVALCamScripts2` | void | `` | Maya_MEL_Proc_Scripts\Circle P |
| `FloatPointsToCamPlane` | float [] | `float $LocPos1[]` | Maya_MEL_Proc_Scripts\MuchBett |
| `getFlyThroughCamera` | string | `` | Maya_MEL_Proc_Scripts\ALL MEL  |
| `GetObjectsInFrontOfPlane2` | string [] | `string $Loc[], vector $VecArray[], vecto...` | Maya_MEL_Proc_Scripts\RADIAL_S |
| `getXrotationOfCam` | float[] | `` | Maya_MEL_Proc_Scripts\Circle P |
| `getYrotationOFcam` | float[] | `` | Maya_MEL_Proc_Scripts\Circle P |
| `IntersectPlaneAndSegment` | int | `vector $Add,vector $p0, vector $N, vecto...` | Maya_MEL_Proc_Scripts\THE_one_ |
| `LineIntersectPlaneCam` | vector [] | `vector $Vecii[], float $CamP[], vector $...` | Maya_MEL_Proc_Scripts\MuchBett |
| `LockModelingCAM` | void | `` | Maya_MEL_Proc_Scripts\Circle P |
| `MirrorPointFrom3PointPlane` | float [] | `vector $SelectVec[], float $oneItemFLoat...` | Maya_MEL_Proc_Scripts\MuchBett |
| `MoveZCURVEModelingCAM` | string[] | `string  $EdgeCurves[], string $ConeLocat...` | Maya_MEL_Proc_Scripts\Circle P |
| `MoveZCURVEModelingCAM2010` | string[] | `string $EdgeCurves[], string $CamConeLoc...` | Maya_MEL_Proc_Scripts\RADIAL_S |
| `MoveZSURFACEModelingCAM` | void | `string  $ConeLocator[]` | Maya_MEL_Proc_Scripts\Circle P |
| `nurbsViewDirectionVectorCam` | float[] | `string $cameraName , int $onlyOrtho` | Maya_MEL_Proc_Scripts\RADIAL_S |
| `PointCurvesToPlaneCurve` | void | `string $CurveItemZ[]` | Maya_MEL_Proc_Scripts\MuchBett |
| `PointCurvesToPlaneCurveB` | string [] | `string $CurveItemZ[]` | Maya_MEL_Proc_Scripts\RADIAL_S |
| `PointInNurbPlane` | string [] | `vector $AllPlaneLocPosition[], string $O...` | Maya_MEL_Proc_Scripts\RADIAL_S |
| `PointToCameraPlane` | void | `string $ObjectLocZx` | Maya_MEL_Proc_Scripts\Workin T |
| `PointToPlaneN` | vector | `vector $Veciiv, vector $Vec[]` | Maya_MEL_Proc_Scripts\MuchBett |
| `ProjectCrv2PlaneNormal` | vector [] | `vector $vecRs[],vector $AvN,vector $Midp` | Maya_MEL_Proc_Scripts\RADIAL_S |
| `ProjectCrv2PlaneNormalP` | vector [] | `vector $vecRs[],vector $AvN,vector $Midp` | Maya_MEL_Proc_Scripts\RADIAL_S |
| `ProjectCrv2PlaneNormalPindex` | vector [] | `vector $vecRs[],vector $AvN,vector $Midp` | Maya_MEL_Proc_Scripts\RADIAL_S |
| `PtDist_to_Plane` | float | `vector $V[],vector $VecN[]` | Maya_MEL_Proc_Scripts\RADIAL_S |
| `queryLocOnLivePlaneBLayer` | string[] | `` | Maya_MEL_Proc_Scripts\RADIAL_S |
| `queryLocOnLivePlaneBLayerDelete` | void | `` | Maya_MEL_Proc_Scripts\RADIAL_S |
| `UnLockModelingCAM` | void | `` | Maya_MEL_Proc_Scripts\Circle P |
| `VecPointsMirrorVecPlaneN` | vector [] | `vector $V[],vector $VecN[]` | Maya_MEL_Proc_Scripts\RADIAL_S |
| `VecPointsToCameraPlane` | vector [] | `vector $V[]` | Maya_MEL_Proc_Scripts\MuchBett |
| `VecPointsToCameraPlaneB` | vector [] | `vector $V[]` | Maya_MEL_Proc_Scripts\RADIAL_S |
| `VecPointsToPlaneX` | vector [] | `vector $V[],vector $n1,vector $p` | Maya_MEL_Proc_Scripts\RADIAL_S |

---

## CURVE Procedures

**SST Layer**: conformal
**Description**: NURBS curves, arc operations

| Procedure | Return | Parameters | Source |
|-----------|--------|------------|--------|
| `addCurveNumbers` | int | `int $addNumbersiA` | Maya_MEL_Proc_Scripts\MuchBett |
| `addStereoCurve` | int | `` | Maya_MEL_Proc_Scripts\MuchBett |
| `AutoCurveScripts` | void | `` | Maya_MEL_Proc_Scripts\Circle P |
| `AutoCurveScripts2` | void | `` | Maya_MEL_Proc_Scripts\Circle P |
| `AutoCurveScriptsTwoCurve` | void | `` | Maya_MEL_Proc_Scripts\MuchBett |
| `averageCurves` | void | `` | Maya_MEL_Proc_Scripts\ALL MEL  |
| `BOUNDRYeveryNthCurve` | void | `` | Maya_MEL_Proc_Scripts\Circle P |
| `calculateEllipseCurve` | vector [] | `vector $FixVeci[], float $steps` | Maya_MEL_Proc_Scripts\MuchBett |
| `CompareCurveIntersect` | string[] | `string  $objectLoc[]` | Maya_MEL_Proc_Scripts\Circle P |
| `CompareCurveIntersect2` | string[] | `string  $object[], string $objectlist[]` | Maya_MEL_Proc_Scripts\Circle P |
| `CompareCurveIntersect4` | string[] | `string  $objectLoc[]` | Maya_MEL_Proc_Scripts\Circle P |
| `CompareCurveIntersect5` | string[] | `string  $objectLoc[], string $objectLocB...` | Maya_MEL_Proc_Scripts\Circle P |
| `CompareCurveIntersectTwoCurves` | string[] | `string $objectLocA[], string $objectAll[...` | Maya_MEL_Proc_Scripts\Circle P |
| `CreateCurve` | string | `float $CurveLength, int  $CurveNSpans, s...` | Maya_MEL_Proc_Scripts\Circle P |
| `CreateCurveFromTwoSelected` | void | `` | Maya_MEL_Proc_Scripts\TANGENTZ |
| `creatStereoCurve` | void | `` | Maya_MEL_Proc_Scripts\MuchBett |
| `curve2points` | string | `float $TanEnd1[], float $intersectposD1[...` | Maya_MEL_Proc_Scripts\Circle P |
| `CurveEPnumber` | int | `string $myCurve` | Maya_MEL_Proc_Scripts\Circle P |
| `CurveExtrude` | void | `` | Maya_MEL_Proc_Scripts\Circle P |
| `CurveIntersectZX` | string | `string $objectcurv[]` | Maya_MEL_Proc_Scripts\Circle P |
| `CutCurveIFConditionTRUETransitZ` | string[] | `string $rebuildit[]` | Maya_MEL_Proc_Scripts\Circle P |
| `DoesCurveBBcrossCurve` | int [] | `string $CrvS,string $CrvAll[]` | Maya_MEL_Proc_Scripts\THE_one_ |
| `DrawCurveDisConnectA` | void | `int $xii` | Maya_MEL_Proc_Scripts\Circle P |
| `DrawCurveDisConnectB` | void | `int $xii` | Maya_MEL_Proc_Scripts\Circle P |
| `EulerAngleofCurve` | float[] | `` | Maya_MEL_Proc_Scripts\Circle P |
| `EvalAddingCurves` | void | `` | Maya_MEL_Proc_Scripts\Circle P |
| `EvalAllCurvesTools` | void | `` | Maya_MEL_Proc_Scripts\Circle P |
| `EvalAllCurvesTools2` | void | `` | Maya_MEL_Proc_Scripts\Circle P |
| `EvalAllCurvesToolZ` | void | `` | Maya_MEL_Proc_Scripts\MuchBett |
| `EvalAutoCurvesScripts` | void | `` | Maya_MEL_Proc_Scripts\Circle P |
| `EvalCurveToolChanged2` | void | `` | Maya_MEL_Proc_Scripts\Circle P |
| `EvalDrawCurveDisConnectA` | void | `` | Maya_MEL_Proc_Scripts\Circle P |
| `EvalDrawCurveDisConnectB` | void | `` | Maya_MEL_Proc_Scripts\Circle P |
| `EvalKilljobAllCurvesToolsNumA` | void | `` | Maya_MEL_Proc_Scripts\Circle P |
| `EvalKilljobAllCurvesToolsNumB` | void | `` | Maya_MEL_Proc_Scripts\Circle P |
| `EvalSmoothCurves` | void | `` | Maya_MEL_Proc_Scripts\Circle P |
| `EvalSmoothCurvesB` | void | `` | Maya_MEL_Proc_Scripts\Circle P |
| `FindclosetTOcurveENDz` | int | `float $a1[], float  $a2[], float $b1[]` | Maya_MEL_Proc_Scripts\Circle P |
| `FindifCurveEndsMeet` | int | `vector $CurveA1[], vector $CurveA2[]` | Maya_MEL_Proc_Scripts\Circle P |
| `FindIfCurveISonZ` | int | `string $rebuildit[]` | Maya_MEL_Proc_Scripts\Circle P |
| `FindIfCurveIsOrthoEpipol` | vector | `vector $EpipolD[], vector $CRV_ENDS[], f...` | Maya_MEL_Proc_Scripts\MuchBett |
| `FindNumOfConnectionsToCurveRemove` | void | `string $boundaryCurves[], int $XiC, stri...` | Maya_MEL_Proc_Scripts\Circle P |
| `FlattenCurveOnMesh` | void | `string $curve[], string  $Mesh[]` | Maya_MEL_Proc_Scripts\Circle P |
| `FlattenCurveToZplain` | void | `string $renamed[]` | Maya_MEL_Proc_Scripts\Circle P |
| `GET_AREA_OF_CURVE` | float | `string $obj[]` | Maya_MEL_Proc_Scripts\Circle P |
| `getCurveLength` | float | `string $curve` | Maya_MEL_Proc_Scripts\Circle P |
| `IfCurvesTouch` | int | `string $curveA, string $curveB` | Maya_MEL_Proc_Scripts\Circle P |
| `ifCurveToolsCTX1` | int | `` | Maya_MEL_Proc_Scripts\Circle P |
| `ifCurveToolsCTX2` | int | `` | Maya_MEL_Proc_Scripts\Circle P |
| `ifCurveToolsCTX3` | int | `` | Maya_MEL_Proc_Scripts\Circle P |
| ... | | | *34 more* |

---

## MATRIX Procedures

**SST Layer**: affine
**Description**: Matrix operations, rotations

| Procedure | Return | Parameters | Source |
|-----------|--------|------------|--------|
| `FloatToMatrix` | matrix | `float $v[]` | Maya_MEL_Proc_Scripts\MuchBett |
| `FloatToMatrixThree` | matrix | `vector $vi[]` | Maya_MEL_Proc_Scripts\RADIAL_S |
| `GetCRV2D_Matrix` | matrix | `vector $PtsVecs[]` | Maya_MEL_Proc_Scripts\THE_one_ |
| `GetInverseMatrix` | matrix | `string $object` | Maya_MEL_Proc_Scripts\MuchBett |
| `GetMatrix` | matrix | `string $object` | Maya_MEL_Proc_Scripts\MuchBett |
| `GetRotationFromDirection` | float [] | `float $P[], float $T[], float $N[]` | Maya_MEL_Proc_Scripts\Workin T |
| `GetRotationVectorsMatrix` | vector [] | `matrix $mAtRiX[][]` | Maya_MEL_Proc_Scripts\MuchBett |
| `getTheTransform` | string | `string $item` | Maya_MEL_Proc_Scripts\ALL MEL  |
| `GetworldInverseMatrix` | matrix | `string $object` | Maya_MEL_Proc_Scripts\MuchBett |
| `GetworldMatrix` | matrix | `string $object` | Maya_MEL_Proc_Scripts\MuchBett |
| `gtExecute3ptSetRotation2` | void | `string $vert2[],string $selectionString[...` | Maya_MEL_Proc_Scripts\complex  |
| `MakeMatrixAxis` | void | `vector $DirectionVector[],vector $MidPt` | Maya_MEL_Proc_Scripts\RADIAL_S |
| `Matrix3D` | void | `` | Maya_MEL_Proc_Scripts\RADIAL_S |
| `Matrix4ToFloat` | float[] | `matrix $m[][]` | Maya_MEL_Proc_Scripts\Circle P |
| `Matrix_Curve_Translation` | vector [] | `vector $VecPairA[],vector $VecPairB[], v...` | Maya_MEL_Proc_Scripts\MuchBett |
| `Matrix_Curve_Translation2D` | vector [] | `vector $VecPairA[],vector $VecPairB[], v...` | Maya_MEL_Proc_Scripts\THE_one_ |
| `Matrix_Curve_TranslationCC` | vector [] | `vector $VecPairA[],vector $VecPairB[], v...` | Maya_MEL_Proc_Scripts\RADIAL_S |
| `MatrixAddCol` | vector | `matrix $mat[][]` | Maya_MEL_Proc_Scripts\MuchBett |
| `MatrixAxisTranlation` | vector | `vector $SVeci, matrix $mIA, matrix $mIB` | Maya_MEL_Proc_Scripts\MuchBett |
| `MatrixCleanNegZero` | matrix | `matrix $m[][]` | Maya_MEL_Proc_Scripts\MuchBett |
| `MatrixDivide` | matrix | `matrix $m[][], matrix $m2[][]` | Maya_MEL_Proc_Scripts\MuchBett |
| `MatrixMirrorX` | matrix | `matrix $m[][]` | Maya_MEL_Proc_Scripts\MuchBett |
| `matrixSpaceVecMult` | vector | `vector $v, matrix $m` | Maya_MEL_Proc_Scripts\MuchBett |
| `MatrixTimesFloat` | matrix | `matrix $m[][], float $X` | Maya_MEL_Proc_Scripts\MuchBett |
| `MatrixToFloat` | float[] | `matrix $FourByFour_matrix[][]` | Maya_MEL_Proc_Scripts\MuchBett |
| `MatrixToFloatN` | matrix | `vector $Vec_matrix[]` | Maya_MEL_Proc_Scripts\RADIAL_S |
| `MatrixToFloatNN` | float[] | `matrix $FourByFour_matrix[][], int $N` | Maya_MEL_Proc_Scripts\RADIAL_S |
| `MultMATRIX` | void | `matrix $rhs[][]` | Maya_MEL_Proc_Scripts\RADIAL_S |
| `MultMatrixMirrorX` | float [] | `float $point[]` | Maya_MEL_Proc_Scripts\MuchBett |
| `MultPointMatrix` | vector | `vector $PtsVec, matrix $mIA[][]` | Maya_MEL_Proc_Scripts\MuchBett |
| `MultPointMatrixArray` | vector [] | `vector $Vec_Array[], matrix $mIA[][],mat...` | Maya_MEL_Proc_Scripts\RADIAL_S |
| `MultPointMatrixPlusRel` | vector | `vector $PtsVec, matrix $mIA[][],matrix $...` | Maya_MEL_Proc_Scripts\RADIAL_S |
| `MultPointMatrixProduct` | vector | `vector $PtsVec, matrix $mIA[][]` | Maya_MEL_Proc_Scripts\MuchBett |
| `needTheLeadTransform` | string | `` | Maya_MEL_Proc_Scripts\ALL MEL  |
| `pointMatrixMult` | float[] | `float $point[], float $matrix[]` | Maya_MEL_Proc_Scripts\MuchBett |
| `PrintMatrix` | void | `matrix $MatriXM[][] , int $MatrixN` | Maya_MEL_Proc_Scripts\MuchBett |
| `RampRotate` | void | `` | Maya_MEL_Proc_Scripts\Workin T |
| `RotateDir_Axis` | vector [] | `float $Rot_T, float $theta, vector $Axis...` | Maya_MEL_Proc_Scripts\RADIAL_S |
| `RotateItX` | void | `float $Rotation[]` | Maya_MEL_Proc_Scripts\TANGENTZ |
| `scaleMATRIX_A` | void | `float $f` | Maya_MEL_Proc_Scripts\RADIAL_S |
| `scaleMATRIX_B` | void | `float $xf, float $yf, float $zf` | Maya_MEL_Proc_Scripts\RADIAL_S |
| `screenSpaceGetMatrix` | matrix | `string $attr` | Maya_MEL_Proc_Scripts\MuchBett |
| `SetGMATRIX` | void | `` | Maya_MEL_Proc_Scripts\RADIAL_S |
| `SetRotation3Points` | void | `` | Maya_MEL_Proc_Scripts\complex  |
| `setRotationAxis` | void | `string $objectitem1[]` | Maya_MEL_Proc_Scripts\Circle P |
| `setRotationAxisFloat` | void | `string $objectitem1[], float $rotationsA...` | Maya_MEL_Proc_Scripts\Circle P |
| `SetRotations` | void | `string $objectSet[], float  $EulerAngleA...` | Maya_MEL_Proc_Scripts\Circle P |
| `SetRotationVectorsMatrix` | matrix | `matrix $mAtRiX[][], float $MfloatRot[]` | Maya_MEL_Proc_Scripts\RADIAL_S |
| `ShowMatrixAXIS` | void | `float $Mn[]` | Maya_MEL_Proc_Scripts\RADIAL_S |
| `StringtoStringMATRIX` | string | `` | Maya_MEL_Proc_Scripts\RADIAL_S |
| ... | | | *19 more* |

---

## ARRAY Procedures

**SST Layer**: affine
**Description**: Array manipulation, sorting, conversion

| Procedure | Return | Parameters | Source |
|-----------|--------|------------|--------|
| `Add_Float_to_3PointFloats` | float[] | `float  $posA, float $posB[]` | Maya_MEL_Proc_Scripts\Circle P |
| `AddFloatArrays` | float [] | `float $FloatArrayA[], float $FloatArrayB...` | Maya_MEL_Proc_Scripts\MuchBett |
| `AddFloats` | float[] | `float $posA[], float  $posB[]` | Maya_MEL_Proc_Scripts\Circle P |
| `AddItemsFromIndexAtoB` | int | `int $Ai, int $Bi, int $numberArrayi[]` | Maya_MEL_Proc_Scripts\RADIAL_S |
| `AddItemsFromIndexAtoBFindZero` | int[] | `int $Ai, int $Bi, int $numberArrayi[]` | Maya_MEL_Proc_Scripts\RADIAL_S |
| `AddItemString` | string | `int $iN, string $NumLetorSy` | Maya_MEL_Proc_Scripts\MuchBett |
| `AppendAllArrays` | string [] | `string $A[] , string $B[]` | Maya_MEL_Proc_Scripts\THE_one_ |
| `AppendArrayZ` | string [] | `string $A[] , string $B[]` | Maya_MEL_Proc_Scripts\RADIAL_S |
| `AppendCrvN_And_Vec` | void | `int $CurveIndex` | Maya_MEL_Proc_Scripts\THE_one_ |
| `AppendFloat` | void | `float $A[] , float $B[]` | Maya_MEL_Proc_Scripts\MuchBett |
| `AppendFloatArray` | void | `float $ExistingF[],float $AddtoExistingF...` | Maya_MEL_Proc_Scripts\MuchBett |
| `AppendFloatsZ` | float[] | `float $posA[], float  $posB[], int $XYZ` | Maya_MEL_Proc_Scripts\Circle P |
| `AppendIntArray` | void | `int $ExistingInt[],int $AddtoExistingInt...` | Maya_MEL_Proc_Scripts\MuchBett |
| `appendMultiStringArray` | void | `string  $copyTo[], string $copyFrom1[], ...` | Maya_MEL_Proc_Scripts\Circle P |
| `ArcLengthArray` | float [] | `string $Objs[]` | Maya_MEL_Proc_Scripts\TANGENTZ |
| `Array_Objects_On_Surface` | void | `int $Unumber, int $Vnumber, string $Obje...` | Maya_MEL_Proc_Scripts\Workin T |
| `ArrayFromAllinString` | string[] | `string  $list` | Maya_MEL_Proc_Scripts\Circle P |
| `ArrayInsertAtEnd` | int | `string $INarray[],  string $NewItem` | Maya_MEL_Proc_Scripts\Circle P |
| `arrayMatch` | int | `string $array[], string $match` | Maya_MEL_Proc_Scripts\ALL MEL  |
| `ArrayNormalVec` | void | `` | Maya_MEL_Proc_Scripts\Workin T |
| `ArrayToIntList` | int[] | `string  $singleStringItemB[]` | Maya_MEL_Proc_Scripts\Circle P |
| `ArrayToVertexFaceNormal` | void | `` | Maya_MEL_Proc_Scripts\Circle P |
| `AverageCurveFloat` | float [] | `float $newVec[]` | Maya_MEL_Proc_Scripts\RADIAL_S |
| `AverageCurveVec` | vector [] | `vector $newVec[]` | Maya_MEL_Proc_Scripts\MuchBett |
| `AverageFloatArrays` | float [] | `float $FloatArrayA[], float $FloatArrayB...` | Maya_MEL_Proc_Scripts\MuchBett |
| `AverageFloats` | float | `float $FloatArray[]` | Maya_MEL_Proc_Scripts\MuchBett |
| `AverageVectorPoint` | float[] | `vector $worldPosZ[]` | Maya_MEL_Proc_Scripts\Circle P |
| `ClosestPoint2LineVec` | vector | `float $DirectionLineF[], float $PointOnL...` | Maya_MEL_Proc_Scripts\MuchBett |
| `ClosestPoint2LineVecX` | vector | `float $DirectionLineF[], float $PointOnL...` | Maya_MEL_Proc_Scripts\RADIAL_S |
| `concatArray` | void | `string $res[], string $in[]` | Maya_MEL_Proc_Scripts\ALL MEL  |
| `convert2Index` | string | `string $Obj` | Maya_MEL_Proc_Scripts\ALL MEL  |
| `CreateIntIndex` | int [] | `int $ArraySize` | Maya_MEL_Proc_Scripts\RADIAL_S |
| `CreateIntIndexExpand` | int [] | `int $IndexArray[], int $ExpandSize` | Maya_MEL_Proc_Scripts\MuchBett |
| `CreateIntIndexF` | float [] | `int $ArraySize, float $Fstep` | Maya_MEL_Proc_Scripts\MuchBett |
| `CreatePairIntIndex` | int [] | `int $AS` | Maya_MEL_Proc_Scripts\MuchBett |
| `CreatePairIntIndexEven` | int [] | `int $ArraySize` | Maya_MEL_Proc_Scripts\MuchBett |
| `CreatePolyFromFloats` | string[] | `float  $BoxPointsX[]` | Maya_MEL_Proc_Scripts\Circle P |
| `CreateVectorInfoAtSelected` | void | `` | Maya_MEL_Proc_Scripts\TANGENTZ |
| `CRVIndexPairF` | int | `int $XNum` | Maya_MEL_Proc_Scripts\THE_one_ |
| `CurveIndexVecTracking` | void | `string $OBJ[]` | Maya_MEL_Proc_Scripts\RADIAL_S |
| `CurveLengthToFloat` | void | `float $Number` | Maya_MEL_Proc_Scripts\TANGENTZ |
| `CycleFloatsZ` | float[] | `float $posA[], float  $posB[], int $XYZ1...` | Maya_MEL_Proc_Scripts\Circle P |
| `CycleNumberString` | string[] | `string  $singleStringItemC[]` | Maya_MEL_Proc_Scripts\Circle P |
| `DiffFloatArrays` | float [] | `float $FloatArrayA[], float $FloatArrayB...` | Maya_MEL_Proc_Scripts\MuchBett |
| `DirectionFString` | float[] | `string $twoItem[]` | Maya_MEL_Proc_Scripts\Circle P |
| `DivideFloatArrays` | float [] | `float $FloatArrayA[], float $DivideX` | Maya_MEL_Proc_Scripts\MuchBett |
| `EvalMoveCurvesTOend` | string[] | `string  $paramANDCurveZ[], string $param...` | Maya_MEL_Proc_Scripts\Circle P |
| `evalVectorIndexAdditionSubtract` | void | `` | Maya_MEL_Proc_Scripts\RADIAL_S |
| `FindEqualVec` | int [] | `vector $VectorA[], vector $Vec` | Maya_MEL_Proc_Scripts\MuchBett |
| `FindifArraysContain` | int | `string  $FirstList[], string $array2[]` | Maya_MEL_Proc_Scripts\Circle P |
| ... | | | *137 more* |

---

## POLYGON Procedures

**SST Layer**: affine
**Description**: Polygon operations, faces, edges

| Procedure | Return | Parameters | Source |
|-----------|--------|------------|--------|
| `addFacesToShard` | string[] | `string $faceList[], int $shardIndex, int...` | Maya_MEL_Proc_Scripts\ALL MEL  |
| `border_edge_err` | void | `` | Maya_MEL_Proc_Scripts\ALL MEL  |
| `checkEdge` | void | `string $Selection[]` | Maya_MEL_Proc_Scripts\ALL MEL  |
| `checkFaces` | void | `string $Selection[]` | Maya_MEL_Proc_Scripts\ALL MEL  |
| `computePolysetVolume` | float | `` | Maya_MEL_Proc_Scripts\ALL MEL  |
| `ConvertPolyFacesIntoNurbSurface` | string | `` | Maya_MEL_Proc_Scripts\Circle P |
| `createRegularPolygonX` | float[] | `int $n,float $r` | Maya_MEL_Proc_Scripts\fixedITC |
| `createRegularPolygonXY` | float[] | `int $n, float $r` | Maya_MEL_Proc_Scripts\True Tan |
| `createRegularPolygonY` | float[] | `int $n, float $r` | Maya_MEL_Proc_Scripts\fixedITC |
| `deleteSingleVertex` | void | `` | Maya_MEL_Proc_Scripts\ALL MEL  |
| `evalEdgecurvesZ2` | void | `` | Maya_MEL_Proc_Scripts\Circle P |
| `FaceEdgeNumber` | int | `string $face` | Maya_MEL_Proc_Scripts\Circle P |
| `facenormal` | float[] | `string $selObj[]` | Maya_MEL_Proc_Scripts\Circle P |
| `fillShardsWithRemainingFaces` | void | `string $object, int $shardIndexList[], i...` | Maya_MEL_Proc_Scripts\ALL MEL  |
| `FindifFourCurvesShareSurface` | int | `string $Foundboundarycurves[]` | Maya_MEL_Proc_Scripts\matrixRo |
| `ForAllCurvesFindSurfaceEdges` | string[] | `string $newallCurves[]` | Maya_MEL_Proc_Scripts\Circle P |
| `GetCenterPointofFace` | float[] | `float  $CurveXYZ[]` | Maya_MEL_Proc_Scripts\Circle P |
| `GetdiffEdgesOfBorder` | string[] | `string  $poly[]` | Maya_MEL_Proc_Scripts\Circle P |
| `GetEdgeVertex` | string[] | `string $poly []` | Maya_MEL_Proc_Scripts\Circle P |
| `GetNearEdges` | string[] | `string  $edgeArray[]` | Maya_MEL_Proc_Scripts\Circle P |
| `getSharedEdges` | string[] | `string $faces[]` | Maya_MEL_Proc_Scripts\ALL MEL  |
| `GetSurfaceIntersectPoint` | float[] | `string $Curvez, string $surf` | Maya_MEL_Proc_Scripts\Circle P |
| `gridSurface` | void | `string $nurb, int $u, int $v` | Maya_MEL_Proc_Scripts\ALL MEL  |
| `InsidePolygon` | int | `vector $c[],vector $VLocPos_A` | Maya_MEL_Proc_Scripts\fixedITC |
| `makePolygonQUAD` | string | `vector $VecPts[]` | Maya_MEL_Proc_Scripts\fixedITC |
| `makeRegularPolygon` | int | `int $n, float $len` | Maya_MEL_Proc_Scripts\ALL MEL  |
| `makeSurfaceAndclose` | void | `string $dialogBoxWin` | Maya_MEL_Proc_Scripts\Circle P |
| `mm_curveFromEdges` | string[] | `` | Maya_MEL_Proc_Scripts\ConvertP |
| `mm_extractCurveFromEdges` | string | `int $degree` | Maya_MEL_Proc_Scripts\ConvertP |
| `mm_extractCurveFromEdgesChooser` | void | `int $degrSel` | Maya_MEL_Proc_Scripts\ConvertP |
| `mm_extractCurveFromEdgesGUI` | void | `` | Maya_MEL_Proc_Scripts\ConvertP |
| `OMT_to_spinEdge` | void | `` | Maya_MEL_Proc_Scripts\ALL MEL  |
| `polyBevelMap` | void | `` | Maya_MEL_Proc_Scripts\ALL MEL  |
| `PolyBorderEdges` | string[] | `string  $polyZ[]` | Maya_MEL_Proc_Scripts\Circle P |
| `polyBridgeFaces` | void | `` | Maya_MEL_Proc_Scripts\ALL MEL  |
| `polyChamferVtx` | string | `int $doHistory, float $width, int $delet...` | Maya_MEL_Proc_Scripts\ALL MEL  |
| `polyCheckSelection` | string[] | `string $fun, string $funtype, int $expan...` | Maya_MEL_Proc_Scripts\ALL MEL  |
| `polyConvertToLoopAndDelete` | void | `` | Maya_MEL_Proc_Scripts\ALL MEL  |
| `polyConvertToLoopAndDuplicate` | void | `` | Maya_MEL_Proc_Scripts\ALL MEL  |
| `polyConvertToRingAndCollapse` | void | `` | Maya_MEL_Proc_Scripts\ALL MEL  |
| `polyConvertToRingAndSplit` | void | `` | Maya_MEL_Proc_Scripts\ALL MEL  |
| `polyConvertToShell` | void | `` | Maya_MEL_Proc_Scripts\ALL MEL  |
| `polyConvertToShellBorder` | void | `` | Maya_MEL_Proc_Scripts\ALL MEL  |
| `polyDeleteVertex` | void | `` | Maya_MEL_Proc_Scripts\ALL MEL  |
| `polySelectEdgesPattern` | void | `string  $method` | Maya_MEL_Proc_Scripts\Circle P |
| `processLeftOverFaces` | int | `string $object, int $shardIndexList[], i...` | Maya_MEL_Proc_Scripts\ALL MEL  |
| `Removefromsurface` | void | `` | Maya_MEL_Proc_Scripts\Circle P |
| `selectTextureToGeomFaces` | string[] | `string $textureToGeom, int $index` | Maya_MEL_Proc_Scripts\ALL MEL  |
| `spFrSurface` | void | `string $nurbsAll[], int $u, int $v` | Maya_MEL_Proc_Scripts\ALL MEL  |
| `subSurface` | string | `` | Maya_MEL_Proc_Scripts\ALL MEL  |
| ... | | | *3 more* |

---

## UTILITY Procedures

**SST Layer**: affine
**Description**: General utilities, selection, cleanup

| Procedure | Return | Parameters | Source |
|-----------|--------|------------|--------|
| `addedNumbers` | int | `int $addNumbersA` | Maya_MEL_Proc_Scripts\Circle P |
| `addNumbers` | int | `` | Maya_MEL_Proc_Scripts\Circle P |
| `AddorSubtract` | int | `int $Number, int $AorS` | Maya_MEL_Proc_Scripts\Circle P |
| `addShaders` | void | `string $textureToGeom, string $mesh` | Maya_MEL_Proc_Scripts\ALL MEL  |
| `addTexture` | void | `string $node, string $file` | Maya_MEL_Proc_Scripts\ALL MEL  |
| `adj_err` | void | `` | Maya_MEL_Proc_Scripts\ALL MEL  |
| `AlphaSpread` | void | `` | Maya_MEL_Proc_Scripts\ALL MEL  |
| `alternativeBoundry` | string | `` | Maya_MEL_Proc_Scripts\Circle P |
| `Angel2Direction` | float | `vector $Da,vector $Db` | Maya_MEL_Proc_Scripts\fixedITC |
| `Angel2DirectionR` | float | `vector $Da,vector $Db` | Maya_MEL_Proc_Scripts\fixedITC |
| `Angle2D` | float | `float $x1, float $y1, float $x2, float $...` | Maya_MEL_Proc_Scripts\fixedITC |
| `angle_to_internal` | float | `float $angle` | Maya_MEL_Proc_Scripts\Circle P |
| `AngleofTwoLines` | float | `vector $vecA,vector $vecB,vector $vecC,v...` | Maya_MEL_Proc_Scripts\MuchBett |
| `AnglesofTriangle` | float [] | `float $SideA, float $SideB, float $SideC` | Maya_MEL_Proc_Scripts\TANGENTZ |
| `appendAll` | void | `string $to[], string $from[]` | Maya_MEL_Proc_Scripts\Circle P |
| `AppendOrdeleteCRV` | void | `int $CRVN[],int $AP` | Maya_MEL_Proc_Scripts\THE_one_ |
| `assembleCmd` | string | `` | Maya_MEL_Proc_Scripts\ALL MEL  |
| `assignShader` | void | `string $cnv, string $shader, int $index` | Maya_MEL_Proc_Scripts\ALL MEL  |
| `AutoBoundryScript` | string[] | `string  $everyFirstCurveShapeset[]` | Maya_MEL_Proc_Scripts\Circle P |
| `averagingNode` | void | `string $firstObjectz, string  $averageOb...` | Maya_MEL_Proc_Scripts\Circle P |
| `avoidZero` | float | `float $N` | Maya_MEL_Proc_Scripts\matrixRo |
| `bad_sel_err` | void | `` | Maya_MEL_Proc_Scripts\ALL MEL  |
| `BBoxInfo2DHW` | vector | `string $Ii,float $H_W[]` | Maya_MEL_Proc_Scripts\THE_one_ |
| `be_plot_locators` | string[] | `string $curve` | Maya_MEL_Proc_Scripts\Workin T |
| `be_plot_no_roll_locators` | void | `string $curve, vector $upVector` | Maya_MEL_Proc_Scripts\Workin T |
| `boundryall` | string[] | `string $everyFirstCurveShapeset[]` | Maya_MEL_Proc_Scripts\matrixRo |
| `BracketFind` | string [] | `string $Find` | Maya_MEL_Proc_Scripts\RADIAL_S |
| `BracketFindPosNeg` | int | `string $Find, int $NegPosZ[]` | Maya_MEL_Proc_Scripts\RADIAL_S |
| `BracketFindPosNegRuleE` | int | `int $CN, string $NewLineX, string $TEXTA...` | Maya_MEL_Proc_Scripts\RADIAL_S |
| `breed` | matrix | `matrix $genomes, float $fitness[], int $...` | Maya_MEL_Proc_Scripts\RADIAL_S |
| `brickwall2` | void | `float $height, float $length, vector $br...` | Maya_MEL_Proc_Scripts\Workin T |
| `calculateEllipse` | vector [] | `float $x, float $y, float $a, float $b, ...` | Maya_MEL_Proc_Scripts\workingc |
| `ClosestPOC` | string | `string $myCurve, string $toObject` | Maya_MEL_Proc_Scripts\Circle P |
| `ClosestPoint2Line` | float [] | `float $DirectionLineF[], float $PointOnL...` | Maya_MEL_Proc_Scripts\RADIAL_S |
| `ClosestPOS` | string | `string $mySurf, string $toObject` | Maya_MEL_Proc_Scripts\Circle P |
| `ConnectMesh` | void | `string $selectOUT[], string  $selectIN[]` | Maya_MEL_Proc_Scripts\Circle P |
| `ConnectMeshScale` | void | `string $selectOUT[], string  $selectIN[]` | Maya_MEL_Proc_Scripts\Circle P |
| `connectObjectToPointOnSubd` | void | `` | Maya_MEL_Proc_Scripts\ALL MEL  |
| `connectObjectToPointOnSubdOne` | void | `string $object, string $face` | Maya_MEL_Proc_Scripts\ALL MEL  |
| `cornerOfTri` | int | `` | Maya_MEL_Proc_Scripts\ALL MEL  |
| `CPerceptron_CPerceptron` | void | `` | Maya_MEL_Proc_Scripts\MuchBett |
| `CPerceptron_Getx0Weight` | float | `` | Maya_MEL_Proc_Scripts\MuchBett |
| `CPerceptron_Getx1Weight` | float | `` | Maya_MEL_Proc_Scripts\MuchBett |
| `CPerceptron_Run` | float | `float $x0, float $x1` | Maya_MEL_Proc_Scripts\MuchBett |
| `CPerceptron_RunX` | float | `float $x0` | Maya_MEL_Proc_Scripts\MuchBett |
| `CPerceptron_SetValues` | void | `float $x0WeightPar, float $x1WeightPar, ...` | Maya_MEL_Proc_Scripts\MuchBett |
| `CPerceptron_Sigmoid` | float | `float $x` | Maya_MEL_Proc_Scripts\MuchBett |
| `CPerceptron_Train` | float | `float $x0, float $x1, float $r` | Maya_MEL_Proc_Scripts\MuchBett |
| `CPerceptron_TrainX` | float | `float $x0, float $x1` | Maya_MEL_Proc_Scripts\MuchBett |
| `CPerceptronZ_RunOUTPUT` | float [] | `float $x0[], float $x1[]` | Maya_MEL_Proc_Scripts\MuchBett |
| ... | | | *202 more* |

---

## Duplicate Procedures (Consolidation Candidates)

These procedures appear in multiple files and should be consolidated:

| Procedure | Occurrences | Files |
|-----------|-------------|-------|
| `GetDistancePointPositionFLOAT` | 17 | Maya_MEL_Proc_Script... |
| `paramToCurvePts3` | 16 | Maya_MEL_Proc_Script... |
| `ArrayInsertAtEnd` | 14 | Maya_MEL_Proc_Script... |
| `MirrorANDrename` | 14 | Maya_MEL_Proc_Script... |
| `stringArrayGmatch` | 12 | Maya_MEL_Proc_Script... |
| `CompareCurveIntersect` | 12 | Maya_MEL_Proc_Script... |
| `GetDistanceBetweenCurveEnds` | 12 | Maya_MEL_Proc_Script... |
| `MidPointBetween` | 12 | Maya_MEL_Proc_Script... |
| `PointsEquivalentTol` | 12 | Maya_MEL_Proc_Script... |
| `SortEvenArrays` | 11 | Maya_MEL_Proc_Script... |
| `paramToCurvePts2` | 11 | Maya_MEL_Proc_Script... |
| `Removefromsurface` | 11 | Maya_MEL_Proc_Script... |
| `EVALCamScripts2` | 11 | Maya_MEL_Proc_Script... |
| `SteinerChain` | 11 | Maya_MEL_Proc_Script... |
| `ArcLengthArray` | 10 | Maya_MEL_Proc_Script... |
| `screenSpaceVecMult` | 10 | Maya_MEL_Proc_Script... |
| `screenSpaceGetMatrix` | 10 | Maya_MEL_Proc_Script... |
| `ConvertPolyFacesIntoNurbSurface` | 10 | Maya_MEL_Proc_Script... |
| `createRegularPolygonX` | 10 | Maya_MEL_Proc_Script... |
| `createRegularPolygonY` | 10 | Maya_MEL_Proc_Script... |
| `CycleNumberString` | 9 | Maya_MEL_Proc_Script... |
| `stringArrayGmatchToArray` | 9 | Maya_MEL_Proc_Script... |
| `Circles_Direction` | 9 | Maya_MEL_Proc_Script... |
| `AddorSubtract` | 9 | Maya_MEL_Proc_Script... |
| `TotalArcLength` | 9 | Maya_MEL_Proc_Script... |
| `PointArray` | 8 | Maya_MEL_Proc_Script... |
| `RoundFloat` | 8 | Maya_MEL_Proc_Script... |
| `CircleFromCurveN` | 8 | Maya_MEL_Proc_Script... |
| `CircleFromCurveT` | 8 | Maya_MEL_Proc_Script... |
| `EulerAngleofCurve` | 8 | Maya_MEL_Proc_Script... |

---

## Key Procedures for SST Integration

### Python Translation Candidates

High-priority procedures for translation to Python `math_core.py`:

| Procedure | Category | Purpose | SST Node |
|-----------|----------|---------|----------|
| `Circle3Point` | circle | Create circle from 3 points - core conformal operation | MutationNode |
| `PointToCircleTangents` | tangent | Calculate tangent lines from point to circle | MutationNode |
| `TangentCircles` | tangent | Find tangent lines between two circles | MutationNode |
| `CircleFromCurve` | circle | Extract circle from NURBS curve | MutationNode |
| `xyzRotation` | matrix | Quaternion-based 3D rotation | TransformNode |
| `GetRotationFromDirection` | matrix | Rotation matrix from direction vectors | TransformNode |
| `ProjectToCameraPlane` | sketch | Project point/curve to camera plane | MutationNode |
| `IntersectTwoCircles` | circle | Circle intersection calculation | MutationNode |

---

## Naming Conventions

| Suffix | Meaning |
|--------|---------|
| `*Z` | Z-plane focused operations |
| `*Vec` | Vector-based operations |
| `*Float` | Float coordinate input |
| `*String` | String array input |
| `*2`, `*3` | Iterative improvements |
| `*TF` | Returns True/False |

---

*Generated by extract_procedures.py and generate_catalog.py*