import numpy as np
import pandas as pd
from scipy.optimize import linprog
from typing import Dict, List, Tuple, Any

class StockManagementSystem:
    def __init__(self, data_path: str):
        """
        Initialize the stock management system with input data
        
        :param data_path: Path to the CSV file containing stock and sales data
        """
        self.df = pd.read_csv(data_path)
        
        # Preprocessing and data validation
        self._validate_data()
        self._preprocess_data()
        
        # System parameters
        self.SAFETY_STOCK_DAYS = 3
        self.MIN_PRODUCTION_HOURS = 2
        self.PRODUCTION_CHANGE_PENALTY_ART = 0.15
        self.PRODUCTION_CHANGE_PENALTY_GROUP = 0.30

    def _validate_data(self):
        """
        Validate input data columns and basic data integrity
        """
        required_columns = [
            'CÓDIGO', 'NOMBRE', 'PRODUCCIÓN HORA', 'LINEA FABRICACIÓN', 
            'STOCK DISPONIBLE MATRIZ', 'STOCK PENDIENTE CALIDAD MATRIZ', 
            'STOCK EXTERNO', 'VENTA 15D', 'SALIDA MEDIA 15D',
            'VENTA AÑO ANTERIOR 15D-', '%COMPARATIVA ACTUAL/ANTERIOR'
        ]
        
        for col in required_columns:
            if col not in self.df.columns:
                raise ValueError(f"Missing required column: {col}")

    def _preprocess_data(self):
        """
        Preprocess and calculate additional features
        """
        # Calculate total stock
        self.df['STOCK_TOTAL'] = (
            self.df['STOCK DISPONIBLE MATRIZ'] + 
            self.df['STOCK PENDIENTE CALIDAD MATRIZ'] + 
            self.df['STOCK EXTERNO']
        )
        
        # Calculate demand estimation
        self.df['DEMANDA_MEDIA'] = self.df['SALIDA MEDIA 15D']
        
        # Apply seasonal adjustment if variation is significant
        mask = np.abs(self.df['%COMPARATIVA ACTUAL/ANTERIOR']) > 20
        self.df.loc[mask, 'DEMANDA_MEDIA'] *= (
            1 + self.df.loc[mask, '%COMPARATIVA ACTUAL/ANTERIOR'] / 100
        )

    def calculate_safety_stock(self) -> Dict[str, float]:
        """
        Calculate safety stock levels for each product
        
        :return: Dictionary of product safety stock levels
        """
        safety_stocks = {}
        for _, row in self.df.iterrows():
            safety_stocks[row['CÓDIGO']] = (
                row['DEMANDA_MEDIA'] * self.SAFETY_STOCK_DAYS
            )
        return safety_stocks

    def optimize_production(
        self, 
        available_hours: float, 
        maintenance_hours: float
    ) -> pd.DataFrame:
        """
        Optimize production using linear programming
        
        :param available_hours: Total available production hours
        :param maintenance_hours: Scheduled maintenance hours
        :return: Optimized production plan
        """
        # Adjust available hours
        net_hours = available_hours - maintenance_hours
        
        # Prepare optimization parameters
        production_hours = self.df['PRODUCCIÓN HORA'].values
        current_stock = self.df['STOCK_TOTAL'].values
        demand = self.df['DEMANDA_MEDIA'].values
        
        # Objective: Minimize stock shortage
        c = -1 * (demand - current_stock)  # Negative to maximize stock coverage
        
        # Constraints
        A_ub = []
        b_ub = []
        
        # Total production hours constraint
        A_ub.append(production_hours)
        b_ub.append(net_hours)
        
        # Non-negativity constraints
        A_eq = np.eye(len(production_hours))
        b_eq = np.zeros(len(production_hours))
        
        # Solve linear programming problem
        result = linprog(
            c, 
            A_ub=np.array(A_ub), 
            b_ub=np.array(b_ub),
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=(0, None),  # Non-negative production
            method='highs'  # Updated to recommended method
        )
        
        # Create production plan DataFrame
        production_plan = self.df.copy()
        production_plan['CAJAS_PRODUCIR'] = result.x
        
        return production_plan

    def generate_production_report(
        self, 
        production_plan: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive production report
        
        :param production_plan: Optimized production plan
        :return: Production report summary
        """
        report = {
            'total_production': production_plan['CAJAS_PRODUCIR'].sum(),
            'products_to_produce': production_plan[
                production_plan['CAJAS_PRODUCIR'] > 0
            ][['CÓDIGO', 'NOMBRE', 'CAJAS_PRODUCIR']].to_dict('records'),
            'potential_stockouts': production_plan[
                production_plan['STOCK_TOTAL'] < 
                production_plan['DEMANDA_MEDIA'] * self.SAFETY_STOCK_DAYS
            ][['CÓDIGO', 'NOMBRE', 'STOCK_TOTAL', 'DEMANDA_MEDIA']].to_dict('records')
        }
        
        return report

def main():
    # Example usage
    stock_system = StockManagementSystem('stock_data.csv')
    
    # Calculate safety stock levels
    safety_stocks = stock_system.calculate_safety_stock()
    
    # Optimize production
    production_plan = stock_system.optimize_production(
        available_hours=240,  # 240 hours per week mentioned in document
        maintenance_hours=10  # Example maintenance hours
    )
    
    # Generate report
    report = stock_system.generate_production_report(production_plan)
    
    print("Production Plan Report:")
    print(report)

if __name__ == "__main__":
    main()